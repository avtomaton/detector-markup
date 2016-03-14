#include "detector-gui.h"

#include <common/stringutils.h>
#include <common/logging.h>
#include <common/profiler.h>

#include "ui_detector-gui.h"

#include <stdint.h>
#include <cstdio>
#include <memory>

#include <fstream>
#include <iostream>
#include <string>
#include <list>

#include <QtCore/QDebug>

namespace ground_truth {

const int DetWindow::CACHE_MAX_FRAMES = 100;
const int DetWindow::JUMP_FRAMES = 500;
const int DetWindow::TRACE_FRAMES_NUM = 15;

static Task task;
static Results results;

using aifil::log_warning;
using aifil::log_state;
using aifil::log_error;

/*
static SizeMap size_map;
static int size_map_point = -1;
*/

QImage qimage_from_opencv_mat(const cv::Mat &mat)
{
	int ch = mat.channels();
	QImage image(mat.cols, mat.rows, QImage::Format_RGB888);
	for (int y = 0; y < mat.rows; ++y)
	{
		const uint8_t* gray = mat.data + mat.step1() * y;
		uchar* out = image.bits() + image.bytesPerLine() * y;
		for (int x = 0; x < mat.cols; ++x)
		{
			*out = *gray; ++out;
			*out = *gray; ++out;
			*out = *gray; ++out;
			gray += ch;
		}
	}
	return image;
}

QRect qrect_from_opencv_rect(const cv::Rect &rect)
{
	return QRect(rect.x, rect.y, rect.width, rect.height);
}

//////////////////////////////////////////////////////////////////////////////////////////

DetWindow::DetWindow(QWidget* parent):
	QWidget(parent),
	cur_frame_num(0), movie_play_mode(PAUSE), media_seq(0),
	/*det_handler(0), simt_det(0), people_counter(0),*/
	is_detections_visible(false),
	is_gt_visible(true),
	is_targets_visible(false),
	is_results_visible(false),
	is_zones_visible(false),
	mlb_down(false), draw_traces(true),
	new_frame_arrived(false), last_computed_frame(0),
	cur_sample_track(0), cur_found_track(0), cur_sample_point(0)
{
	workmode = MODE_CORRECT_CREATE;
	//visualize_bg = 0;

	last_gt_point.width = 10.0f;
	last_gt_point.height = 20.0f;
	last_gt_point.center_x = 50.0f;
	last_gt_point.center_y = 50.0f;

	edited_track = -1;
	edited_type = 1;

	setMouseTracking(true);
	setFocusPolicy(Qt::StrongFocus);
	setFocus();

	timer = new QTimer(this);
	if (task.wanted_fps)
		timer->setInterval(int(1000 / task.wanted_fps));
	else
		timer->setInterval(40); //25fps by default
	connect(timer, SIGNAL(timeout()), this, SLOT(timeout()));

	active_zone = -1;
}

DetWindow::~DetWindow()
{
	/*
	delete det_handler;
	delete people_counter;
	delete simt_det;
	*/
	delete media_seq;
}

void DetWindow::reset_all()
{
	if (task.is_movie())
		reset(task.movie_path, task.wanted_fps);
	else
		reset(task.photo_dir);
	get_frame(1);

	is_rect_move = false;
	mouse_stretch_mode = NONE;
}

void DetWindow::reset(const std::string &path, double wanted_fps)
{
	set_play_mode(PAUSE);
	cur_frame_num = 0;
	delete media_seq;
	frame_archive.clear();

	fps = wanted_fps;
	address = path;
	timer->setInterval(int(1000.0 / wanted_fps)); //in msec
	media_seq = new SequentalReader;
	media_seq->setup(path, int(wanted_fps));
}

void DetWindow::set_play_mode(PLAY_MODE new_mode)
{
	if (new_mode == PLAY_PAUSE)
	{
		if (movie_play_mode == PAUSE)
			new_mode = PLAY;
		else //PLAY, PLAY_FAST
			new_mode = PAUSE;
	}
	movie_play_mode = new_mode;
	if (new_mode == PAUSE)
		timer->stop();
	else
		timer->start();
}

void DetWindow::ensure_visible(int frame_first, int frame_last)
{
	if (frame_first <= cur_frame_num && cur_frame_num <= frame_last)
		return;

	get_frame(frame_first - cur_frame_num);
}

void DetWindow::read_cur_frame()
{
	cv::Mat mat = media_seq->get_image_gray();
	af_assert(!mat.empty() && "we must have frame here");
	frame_archive[media_seq->cur_frame_num] = qimage_from_opencv_mat(mat);
}

bool DetWindow::get_frame(int frames_from_cur)
{
	cur_sample_point = 0;

	if (!media_seq)
	{
		aifil::log_warning("don't have media");
		return false;
	}

	bool have_next = true;
	int wanted_frame = cur_frame_num + frames_from_cur;
	if (wanted_frame < 1)
	{
		have_next = false;
		wanted_frame = 1;
	}

	if (cur_frame_num == wanted_frame)
		return have_next;


	if (abs(wanted_frame - media_seq->cur_frame_num) >= CACHE_MAX_FRAMES)
		frame_archive.clear();

	bool have_in_archive;
	if (
		frame_archive.empty() ||
		wanted_frame > frame_archive.rbegin()->first ||
		wanted_frame < frame_archive.begin()->first
	)
		have_in_archive = false;
	else
		have_in_archive = true;


	if (have_in_archive)
		cur_frame_num = wanted_frame;
	else
	{
		if (wanted_frame < media_seq->cur_frame_num)
			reset(address, fps);

		while (media_seq->cur_frame_num < wanted_frame)
		{
			if (wanted_frame - media_seq->cur_frame_num <= CACHE_MAX_FRAMES)
			{
				if (!media_seq->feed_frame())
				{
					set_play_mode(PAUSE);
					break;
				}
			}
			else
			{
				if (!media_seq->fast_feed_frame())
				{
					set_play_mode(PAUSE);
					read_cur_frame();
					break;
				}
				else
					continue;
			}

			if (frame_archive.empty() ||
				media_seq->cur_frame_num > frame_archive.rbegin()->first ||
				media_seq->cur_frame_num < frame_archive.begin()->first
			)
				read_cur_frame();
		}
		cur_frame_num = media_seq->cur_frame_num;
		while (frame_archive.size() > CACHE_MAX_FRAMES)
			frame_archive.erase(frame_archive.begin());
	}

	have_next &= (cur_frame_num == wanted_frame);

	if (have_next && frames_from_cur == 1 && cur_frame_num > last_computed_frame)
		compute_frame();
	frame_pixmap = QPixmap::fromImage(frame_archive[cur_frame_num], Qt::ColorOnly);
	QString frame_label = QString("frame: %1").arg(cur_frame_num);
	if (!task.is_movie())
		frame_label += QString(" (%1)").arg(frame_names[cur_frame_num]);
	emit frame_num_changed(frame_label);
	update();
	return have_next;
}

void DetWindow::next_frame()
{
	set_play_mode(PAUSE);
	get_frame(1);
}

void DetWindow::prev_frame()
{
	set_play_mode(PAUSE);
	get_frame(-1);
}

void DetWindow::start_movie()
{
	set_play_mode(PAUSE);
	get_frame(INT_MIN);
	emit ready();
}

void DetWindow::end_movie()
{
	set_play_mode(PAUSE);
	while (get_frame(1))
		;
	emit ready();
}

void DetWindow::jump_frame()
{
	set_play_mode(DetWindow::PAUSE);
	get_frame(JUMP_FRAMES);
}

QString DetWindow::cur_frame_name()
{
	if (media_seq->movie)
		return QString::number(cur_frame_num);
	else
		return frame_names[cur_frame_num];
}

int DetWindow::cur_frame_w()
{
	return frame_archive[cur_frame_num].width();
}

int DetWindow::cur_frame_h()
{
	return frame_archive[cur_frame_num].height();
}

void DetWindow::zones_update()
{
	//if (det_handler)
	//	det_handler->zones_set(zones);
}

void DetWindow::paintEvent(QPaintEvent *)
{
	if (frame_pixmap.isNull())
		return;

	QPainter painter(this);
	painter.drawPixmap(rect(), frame_pixmap);

	QPainter &p = painter;

	int widget_w = rect().width();
	int widget_h = rect().height();

	/*if (is_detections_visible && det_handler)
	{
		painter.save();
		painter.setPen(QPen(Qt::magenta, 1));
		painter.setBrush(QBrush());
		for (int idx = 0; idx < (int)det_handler->cur_detections.size(); ++idx)
		{
			const ResultDetection &t = det_handler->cur_detections[idx];
			QRect r = qrect_from_opencv_rect(t.rect(widget_w, widget_h));
			painter.drawRect(r);
			painter.drawText(r.left() + r.width() / 2, r.top() + r.height() / 5,
				QString::number(t.id));
		}
		painter.restore();
	}*/

	/*if (is_targets_visible && det_handler)
	{
		painter.save();
		for (int idx = 0; idx < (int)det_handler->cur_tracks.size(); ++idx)
		{
			const ResultTarget &t = det_handler->cur_tracks[idx];
			QColor color;
			if (t.ready)
				color = Qt::darkCyan;
			else if (t.confidence < 0)
				color = Qt::yellow;
			else
				color = QColor(0, int(255 * t.confidence), int(255 * t.confidence));

			int x = int(t.center_x * widget_w / 100);
			int y = int(t.center_y * widget_h / 100);
			int w = int(t.width * widget_w / 100);
			int h = int(t.height * widget_h / 100);
			painter.setPen(QPen(color, 1));
			painter.setBrush(QBrush());
			painter.drawRect(x - w / 2, y - h / 2, w, h);
			painter.drawText(x, y, QString::fromUtf8(t.label.c_str()));

			for (std::list<ResultTrack::Point>::const_iterator pit = t.track.points.begin();
				 pit != t.track.points.end(); ++pit)
			{
				std::list<ResultTrack::Point>::const_iterator npit = pit;
				++npit;
				if (npit == t.track.points.end())
					break;

				painter.drawLine(int(pit->x * widget_w / 100),
					int(pit->y * widget_h / 100),
					int(npit->x * widget_w / 100),
					int(npit->y * widget_h / 100));
			}
		}
		painter.restore();
	}*/

	if (is_gt_visible)
	{
		for (tracks_t::iterator it = gt_tracks.begin(); it != gt_tracks.end(); ++it)
		{
			draw_track(
				it->second, &p,
				cur_sample_track && it->first == cur_sample_track->object_id
			);
		}

		/*
		std::vector<GroundTruthPoint*> vec;
		if (cur_frame_num)
			vec = results.all_points_correct.select_by_frame(cur_frame_num);
		else if (!media_seq->cur_frame_name.empty())
			vec = results.all_points_correct.select_by_frame(media_seq->cur_frame_name);
		draw_logic_rect(vec, 0, &painter, Qt::blue, Qt::red);
		*/
	}

	if (is_results_visible)
	{
		/*
		for (tracks_t::iterator it = object_tracks_found.begin(); it != object_tracks_found.end(); ++it)
			draw_track(it->second, &p, cur_obj_track_found && it->first == cur_obj_track_found->object_id);
		*/

		std::vector<GroundTruthPoint*> vec;
		if (task.is_movie())
			vec = results.all_points_detected.select_by_frame(cur_frame_num);
		else
			vec = results.all_points_detected.select_by_frame(
						frame_names[cur_frame_num].toUtf8().data());

		draw_logic_rect(vec, 0, &painter, Qt::green, Qt::darkGreen);
	}

	if (is_zones_visible || (workmode & MODE_ZONE_CREATE))
	{
		painter.save();
		//FIXME: zone coords
		for (size_t i = 0; i < zones.size(); ++i)
		{
			const DetectorZoneParams &zone = zones[i];
			if (zone.points.size() < 4)
				continue;
			QPolygon polygon;
			for (size_t j = 0; j < zone.points.size() / 2; ++j)
				polygon.putPoints(j, 1,
					int(zone.points[2 * j] * widget_w / 100),
					int(zone.points[2 * j + 1] * widget_h / 100));
			painter.setPen(QPen(Qt::cyan, 1));
			if (zone.type == "border" || zone.type == "border_swapped")
				painter.drawPolyline(polygon);
			else
				painter.drawPolygon(polygon);
		}
		painter.restore();
	} //if zones vis
	/*
	if ((workmode & MODE_GROUND_PLANE) && size_map.points.size() == 4)
	{
		painter.save();
		painter.setBrush(QBrush());
		Qt::GlobalColor color0 = Qt::yellow;
		Qt::GlobalColor color1 = Qt::blue;
		Qt::GlobalColor color2 = Qt::red;


		const SizeMap &m = size_map;
		for (int i = 0; i < 4; ++i)
		{
			const SizeMap::SupportPoint &p0 = m.points[i];
			const SizeMap::SupportPoint &p1 = m.points[i == 3 ? 0 : i + 1];
			int x0 = int(p0.x * widget_w / 100);
			int y0 = widget_h - int(p0.y * widget_h / 100);
			int x1 = int(p1.x * widget_w / 100);
			int y1 = widget_h - int(p1.y * widget_h / 100);
			int min_w = int(p0.min_w * widget_w / 100);
			int min_h = int(p0.min_h * widget_h / 100);
			int max_w = int(p0.max_w * widget_w / 100);
			int max_h = int(p0.max_h * widget_h / 100);

			painter.setPen(QPen(color0, 1));
			painter.drawLine(x0, y0, x1, y1);

			int border = 5;
			painter.setPen(QPen(color1, 1));
			int rect_x = std::min(std::max(border, x0 - min_w / 2), widget_w - min_w - border);
			int rect_y = std::min(std::max(border, y0 - min_h), widget_h - min_h - border);
			painter.drawRect(rect_x, rect_y, min_w, min_h);

			painter.setPen(QPen(color2, 1));
			rect_x = std::min(std::max(border, x0 - max_w / 2), widget_w - max_w - border);
			rect_y = std::min(std::max(border, y0 - max_h), widget_h - max_h - border);
			painter.drawRect(rect_x, rect_y, max_w, max_h);
		}

		painter.restore();
	} //if ground plane
	*/
}

void DetWindow::mousePressEvent(QMouseEvent *ev)
{
	if (ev->buttons() != Qt::LeftButton)
	{
		mlb_down = false;
		return;
	}

	mlb_down = true;
	set_play_mode(PAUSE);

	if (gt_tracks.empty())
		return;

	float x_logic = ev->x() * 100.0f / rect().width();
	float y_logic = ev->y() * 100.0f / rect().height();

	mouse_x = int(x_logic);
	mouse_y = int(y_logic);

	old_pos = ev->pos();

	QRect myrect;
	tracks_t::iterator trit = gt_tracks.end();
	for (tracks_t::iterator it = gt_tracks.begin();
		it != gt_tracks.end();
		++it)
	{
		myrect = get_rect_on_widget(it->second, cur_frame_num);
		if (!myrect.isEmpty() && myrect.contains(old_pos))
		{
			trit = it;
			cur_sample_track = &it->second;
			emit gt_track_selected();
			break;
		}
	}

	if (trit == gt_tracks.end())
		return;

	bool clone_track = false;
	if (ev->modifiers() & Qt::ControlModifier)
	{
		clone_track = true;
		add_sample_track(cur_sample_track->object_type,
						 cur_sample_track->get_point(cur_frame_num));
	}

	int x = myrect.x();
	int y = myrect.y();
	int w = myrect.width();
	int h = myrect.height();

	int border_w = std::min(w / 4, 30);
	int border_h = std::min(h / 4, 30);
	QRect move_area(x + border_w, y + border_h, w - 2 * border_w, h - 2 * border_h);
	if (move_area.contains(old_pos) || clone_track)
	{
		is_rect_move = true;
		old_pos_dist = old_pos - QPoint(x + w / 2, y + h / 2);
	}
	else
	{
		if (old_pos.x() < move_area.x())
		{
			if (old_pos.y() < move_area.y())
				mouse_stretch_mode = LEFT_TOP;
			else if (old_pos.y() > move_area.y() + move_area.height())
				mouse_stretch_mode = LEFT_BOTTOM;
			else
				mouse_stretch_mode = LEFT;
		}
		else if(old_pos.x() > move_area.x() + move_area.width())
		{
			if (old_pos.y() < move_area.y())
				mouse_stretch_mode = RIGHT_TOP;
			else if(old_pos.y() > move_area.y() + move_area.height())
				mouse_stretch_mode = RIGHT_BOTTOM;
			else
				mouse_stretch_mode = RIGHT;
		}
		else
		{
			if (old_pos.y() < move_area.y())
				mouse_stretch_mode = TOP;
			else
				mouse_stretch_mode = BOTTOM;
		}
	}

	visualize();
	update();
}

void DetWindow::mouseReleaseEvent(QMouseEvent *ev)
{
	if (ev->button() != Qt::LeftButton)
		return;

	set_play_mode(PAUSE);

	if (mlb_down && (mouse_stretch_mode != NONE || is_rect_move))
		update_current_point();

	mouse_stretch_mode = NONE;
	is_rect_move = false;
	mlb_down = false;

	/*
	if (workmode & MODE_TRACK)
	{
		cv::Rect rectt;
		rectt.x = mouse_x;
		rectt.y = mouse_y;
		rectt.width = (int(x_logic) - mouse_x);
		rectt.height = (int(y_logic) - mouse_y);
		if (rectt.width <= 0 || rectt.height <= 0)
		{
			log_state("wrong rect");
			return;
		}
		if (simt_det)
			simt_det->add_point(rectt);
		if (people_counter)
			people_counter->reset_tracker(rectt);
	}
	else if (workmode & MODE_CROP && people_counter)
		people_counter->setup_area(mouse_x, mouse_y, x_logic, y_logic);
	else if (workmode & MODE_CLASSIFY && people_counter)
		people_counter->cl_score(mouse_x, mouse_y, x_logic, y_logic);
	else if (workmode & MODE_GROUND_PLANE && people_counter && size_map.points.size() == 4)
	{
		people_counter->set_detector_ground_plane(size_map.to_string());
		std::string tres = size_map.to_string();
		FILE *f = fopen("ground_plane", "wb");
		fprintf(f, "%s", tres.c_str());
		fclose(f);
		int pos = 0;
		while ((pos = tres.find_first_of("\n")) > 0)
			tres.replace(pos, 1, 1, ',');
		f = fopen("ground_plane_", "wb");
		fprintf(f, "%s", tres.c_str());
		fclose(f);
	}
	*/

	visualize();
	update();
}

void DetWindow::mouseMoveEvent(QMouseEvent *ev)
{
	float x_logic = ev->x() * 100.0f / rect().width();
	float y_logic = ev->y() * 100.0f / rect().height();

	if (mlb_down)
	{
		if (workmode & MODE_CORRECT_CREATE)
		{
			QPoint pos = ev->pos();
			int dx = pos.x() - old_pos.x();
			int dy = pos.y() - old_pos.y();
			old_pos = pos;
			if (is_rect_move)
				move_rect(dx, dy, false);
			else if (mouse_stretch_mode != NONE)
				stretch_rect(dx, dy, false);
		}
		/*
		else if ((workmode & MODE_GROUND_PLANE) && size_map_point != -1)
		{
			size_map.points[size_map_point].x = x_logic;
			size_map.points[size_map_point].y = 100.0f - y_logic;
		}
		*/

		visualize();
		update();
	}

	std::string mouse_label_text =
		aifil::stdprintf("current coord: %.2f, %.2f"/* (%.2f, %.2f)"*/,
			x_logic, y_logic/*, (float)x_frame, (float)y_frame*/);
	emit mouse_label_changed(QString::fromLatin1(mouse_label_text.c_str()));
}

/*
void DetWindow::update_size_map_point(bool horizontal, bool minimal, double delta)
{

	float x_logic = mapFromGlobal(cursor().pos()).x() * 100.0f / rect().width();
	float y_logic = mapFromGlobal(cursor().pos()).y() * 100.0f / rect().height();

	if ((workmode & MODE_GROUND_PLANE) && size_map.points.size() == 4)
	{
		size_map_point = size_map.nearest_point(x_logic, 100.0f - y_logic);
		if (size_map_point != -1)
		{
			if (horizontal && minimal)
				size_map.points[size_map_point].min_w += delta;
			else if (horizontal && !minimal)
				size_map.points[size_map_point].max_w += delta;
			else if (!horizontal && minimal)
				size_map.points[size_map_point].min_h += delta;
			else if (!horizontal && !minimal)
				size_map.points[size_map_point].max_h += delta;
			visualize();
			update();
		}
	}

}
*/

void DetWindow::kev(QKeyEvent *)
{
	/*
	if ((ev->key() >= Qt::Key_0 && ev->key() <= Qt::Key_9) || ev->key() == Qt::Key_QuoteLeft)
	{
		if (ev->key() == Qt::Key_QuoteLeft)
			visualize_bg = 0;
		else if (ev->key() == Qt::Key_0)
			visualize_bg = 10;
		else
			visualize_bg = ev->key() - Qt::Key_0;
	}
	*/

	picture_settings_update();
}

GUITrack DetWindow::make_track(const GroundTruthPoint &pt)
{
	GUITrack::INTERPOLATION interp = GUITrack::INTERP_LINEAR;
	if (task.interpolation == "polynom")
		interp = GUITrack::INTERP_POLYNOM;
	return GUITrack(pt, interp);
}

void DetWindow::update_result_point(int obj_id, int obj_type,
	float new_x, float new_y, float new_w, float new_h)
{
	if (obj_id == -1)
		return;

	std::string frame_name;
	if (!media_seq->cur_frame_name.empty())
		frame_name = media_seq->cur_frame_name;

	auto rit = results.all_points_detected.update_point(
		cur_frame_num, frame_name, obj_id, obj_type, new_x, new_y, new_w, new_h);

	// object_tracks_found
	rit->frame_num = cur_frame_num;
	tracks_t::iterator itrack = found_tracks.find(obj_id);
	if (itrack == found_tracks.end())
	{

		std::pair<tracks_t::iterator, bool> par =
			found_tracks.insert(std::make_pair(obj_id, make_track(*rit)));
		itrack = par.first;
	}
	emit detected_track_changed();
}

void DetWindow::results2tracks()
{
	gt_tracks.clear();
	found_tracks.clear();
	cur_sample_track = 0;
	cur_found_track = 0;

	QMap<QString, int> frame_numbers;
	if (!task.is_movie())
	{
		for (auto it = frame_names.begin(); it != frame_names.end(); ++it)
			frame_numbers[it.value()] = it.key();
	}

	for (auto it = results.all_points_correct.points.begin();
		it != results.all_points_correct.points.end();
		++it)
	{
		if (!it->is_base)
			continue;

		if (!task.is_movie())
			it->frame_num = frame_numbers.value(it->pic_name.c_str(), 0);
		tracks_t::iterator itrack = gt_tracks.find(it->object_id);
		if (itrack == gt_tracks.end())
		{
			std::pair<tracks_t::iterator, bool> par =
				gt_tracks.insert(std::make_pair(
					it->object_id, make_track(*it)));
			itrack = par.first;
		}
		else
			itrack->second.update_base_pt(*it);
	}

	for (auto it = results.all_points_detected.points.begin();
		it != results.all_points_detected.points.end();
		++it)
	{
		tracks_t::iterator itrack = found_tracks.find(it->object_id);
		if (itrack == found_tracks.end())
		{
			std::pair<tracks_t::iterator, bool> par =
				found_tracks.insert(std::make_pair(
					it->object_id, make_track(*it)));
			itrack = par.first;
		}

		itrack->second.update_base_pt(*it);
	}

	for (tracks_t::iterator it = gt_tracks.begin(); it != gt_tracks.end(); ++it)
		it->second.update();

	for (tracks_t::iterator it = found_tracks.begin(); it != found_tracks.end(); ++it)
		it->second.update();
}

void DetWindow::convert_name2int()
{
	if (task.is_movie())
		return;

	reset_all();
	frame_names.clear();
	do
		frame_names[media_seq->cur_frame_num] = media_seq->cur_frame_name.c_str();
	while (media_seq->fast_feed_frame());
	reset_all();
}

void DetWindow::del_current_track()
{
	if (!cur_sample_track)
		return;

	auto it = gt_tracks.find(cur_sample_track->object_id);
	af_assert(it != gt_tracks.end());
	int removed_id = it->first;
	int row_in_gui_table = it->second.table_row;
	log_state("del current track with id '%d'", removed_id);

	for (auto pt = it->second.points.begin(); pt != it->second.points.end(); ++pt)
		results.all_points_correct.erase_point(pt->second);

	gt_tracks.erase(it);
	if (gt_tracks.empty())
		cur_sample_track = 0;
	else
		cur_sample_track = &(gt_tracks.begin()->second);
	emit gt_track_removed(removed_id, row_in_gui_table);
}

void DetWindow::del_current_point()
{
	log_state("del_current_point");
	if (!cur_sample_track)
		return;

	auto pt = cur_sample_track->points.find(cur_frame_num);
	if (pt == cur_sample_track->points.end() || !pt->second.is_base)
		return;

	results.all_points_correct.erase_point(pt->second);

	cur_sample_track->points.erase(pt);
	if (cur_sample_track->points_base())
	{
		cur_sample_track->update();
		emit gt_track_changed(cur_sample_track->object_id);
	}
	else
		del_current_track();
}

GroundTruthPoint* DetWindow::get_current_point()
{
	if (!cur_sample_track)
		return 0;

	GroundTruthPoint *ob = 0;
	if (cur_sample_track->start_frame <= cur_frame_num &&
		cur_frame_num <= cur_sample_track->finish_frame)
	{
		auto it = cur_sample_track->points.find(cur_frame_num);
		af_assert(it != cur_sample_track->points.end());
		ob = &it->second;
		if (!ob->is_base)
		{
			cur_sample_track->update_base_pt(*ob);

			// point container was updated - update pointer
			it = cur_sample_track->points.find(cur_frame_num);
			af_assert(it != cur_sample_track->points.end());
			ob = &it->second;
			results.all_points_correct.update_point(*ob);
		}
	}
	else if (cur_sample_track->extrapolation_enable)
	{
		GroundTruthPoint p = cur_sample_track->extrapolate(cur_frame_num);
		cur_sample_track->update_base_pt(p);
		auto it = cur_sample_track->points.find(cur_frame_num);
		af_assert(it != cur_sample_track->points.end());
		ob = &it->second;
		results.all_points_correct.update_point(*ob);
	}
	return ob;
}

void DetWindow::update_current_point()
{
	if (!cur_sample_point || !cur_sample_track)
		return;

	cur_sample_track->update_base_pt(*cur_sample_point);
	results.all_points_correct.update_point(*cur_sample_point);
	last_gt_point = *cur_sample_point;
	cur_sample_point = 0;
	emit gt_track_changed(cur_sample_track->object_id);
	update();
}

void DetWindow::move_rect(int dx, int dy, bool update_lists)
{
	if (!cur_sample_point)
		cur_sample_point = get_current_point();
	if (!cur_sample_point)
		return;

	cur_sample_point->center_x +=  100 * dx / float(rect().width());
	cur_sample_point->center_y += 100 * dy / float(rect().height());
	if (update_lists)
		update_current_point();
	else
		update();
}

void DetWindow::stretch_with_mode(int dx, int dy, GroundTruthPoint *ob)
{
	float widget_w = float(rect().width());
	float widget_h = float(rect().height());
	switch (mouse_stretch_mode)
	{
	case NONE:
		ob->width += 100 * dx / widget_w;
		ob->height += 100 * dy / widget_h;
		break;
	case LEFT_TOP:
		ob->center_x += 50 * dx / widget_w;
		ob->center_y += 50 * dy / widget_h;
		ob->width -= 100 * dx / widget_w;
		ob->height -= 100 * dy / widget_h;
		break;
	case TOP:
		ob->center_y += 50 * dy / widget_h;
		ob->height -= 100 * dy / widget_h;
		break;
	case RIGHT_TOP:
		ob->center_x += 50 * dx / widget_w;
		ob->center_y += 50 * dy / widget_h;
		ob->width += 100 * dx / widget_w;
		ob->height -= 100 * dy / widget_h;
		break;
	case LEFT:
		ob->center_x += 50 * dx / widget_w;
		ob->width -= 100 * dx / widget_w;
		break;
	case RIGHT:
		ob->center_x += 50 * dx / widget_w;
		ob->width += 100 * dx / widget_w;
		break;
	case LEFT_BOTTOM:
		ob->center_x += 50 * dx / widget_w;
		ob->center_y += 50 * dy / widget_h;
		ob->width -= 100 * dx / widget_w;
		ob->height += 100 * dy / widget_h;
		break;
	case BOTTOM:
		ob->center_y += 50 * dy / widget_h;
		ob->height += 100 * dy / widget_h;
		break;
	case RIGHT_BOTTOM:
		ob->center_x += 50 * dx / widget_w;
		ob->center_y += 50 * dy / widget_h;
		ob->width += 100 * dx / widget_w;
		ob->height += 100 * dy / widget_h;
		break;
	}

	if (ob->width < 100 * 10 / widget_w)
		ob->width = 100 * 10 / widget_w;
	if (ob->height < 100 * 10 / widget_h)
		ob->height = 100 * 10 / widget_h;
}

void DetWindow::stretch_rect(int dx, int dy, bool update_lists)
{
	if (!cur_sample_point)
		cur_sample_point = get_current_point();
	if (!cur_sample_point)
		return;

	stretch_with_mode(dx, dy, cur_sample_point);
	if (update_lists)
		update_current_point();
	else
		update();
}

void DetWindow::change_cur_point_size(float w, float h)
{
	if (!cur_sample_point)
		cur_sample_point = get_current_point();
	if (!cur_sample_point)
		return;

	cur_sample_point->width = w;
	cur_sample_point->height = h;
	update_current_point();
}

void DetWindow::add_sample_track(int obj_type, const GroundTruthPoint &pt)
{
	log_state("adding new track");
	GroundTruthPoint ob = pt;
	ob.object_type = obj_type;
	ob.is_base = true;
	if (pt.empty())
	{
		ob.center_x = last_gt_point.center_x;
		ob.center_y = last_gt_point.center_y;
		ob.width = last_gt_point.width;
		ob.height = last_gt_point.height;
	}
	int new_id = 1;
	for (auto it = gt_tracks.begin(); it != gt_tracks.end(); ++it)
	{
		if (new_id < it->first)
			break;
		else
			++new_id;
	}
	af_assert(gt_tracks.find(new_id) == gt_tracks.end());
	ob.object_id = new_id;
	ob.frame_num = cur_frame_num;
	if (!task.is_movie())
		ob.pic_name = frame_names[cur_frame_num].toUtf8().data();

	aifil::log_state("adding new track (ID %d)\n", new_id);

	std::pair<tracks_t::iterator, bool> par =
		gt_tracks.insert(std::make_pair(new_id, make_track(ob)));
	GUITrack &track = par.first->second;
	track.extrapolation_enable = task.is_movie();
	results.all_points_correct.update_point(ob);
	cur_sample_track = &track;
	emit gt_track_added(new_id);
	update();
}

QRect DetWindow::get_rect_on_widget(const GUITrack &track, int frame)
{
	int widget_w = rect().width();
	int widget_h = rect().height();
	const GroundTruthPoint &ob = track.get_point(frame);
	if (!ob.empty())
		return qrect_from_opencv_rect(ob.rect(widget_w, widget_h));
	else
		return QRect();
}

void DetWindow::picture_settings_update()
{
	new_frame_arrived = true;
	visualize();
	update();
}

void DetWindow::compute_frame()
{
	if (cur_frame_num != media_seq->cur_frame_num)
	{
		aifil::log_warning("cannot compute historical frame");
		return;
	}

	last_computed_frame = cur_frame_num;
	//if (!det_handler)
		return;
	/*
	det_handler->run();
	if (!det_handler->cur_detections.empty())
		results.add_detections(det_handler->cur_detections, cur_frame_num);
	if (!det_handler->cur_tracks.empty())
		results.add_tracks(det_handler->cur_tracks, cur_frame_num);
	*/
	new_frame_arrived = true;

#ifdef USE_PROFILER
	aifil::log_state("stat:\n%s", Profiler::instance().print_statistics().c_str());
#endif
}


void DetWindow::visualize()
{
	/*int w = 0;
	int h = 0;
	int rs = 0;
	if (simt_det && new_frame_arrived)
	{
		uint8_t *img_bytes = 0;
		simt_det->get_visualization(&w, &h, &rs, &img_bytes, 0, 0, 1);
		QImage img(img_bytes, w, h, rs, QImage::Format_RGB888);
		frame_pixmap = QPixmap::fromImage(img.rgbSwapped());
	}
	else if (media_seq->cooker && new_frame_arrived && people_counter)
	{
		int working_w = people_counter->frame_w;
		int working_h = people_counter->frame_h;
		QImage img = QImage(working_w, working_h, QImage::Format_RGB888);
		const uint8_t *img_bytes = media_seq->cooker->get_bytes_gray(working_w, working_h);
		for (int y = 0; y < working_h; ++y)
		{
			const uint8_t* gray = img_bytes + working_w * y;
			uchar* out = img.bits() + img.bytesPerLine() * y;
			for (int x = 0; x < working_w; ++x)
			{
				*out = *gray; ++out;
				*out = *gray; ++out;
				*out = *gray; ++out;
				++gray;
			}
		}

		if (visualize_bg > 0 && people_counter->detector &&
			people_counter->detector->scales.size() &&
			people_counter->detector->scales.front().images->icf_channels_mat_ready)
		{
			std::shared_ptr<MatCache> sc = people_counter->detector->scales.front().images;
			img = img.scaled(sc->width, sc->height);
			std::vector<cv::Mat> tmp;
			cv::split(sc->icf_channels_mat, tmp);
			for (int y = 0; y < sc->height; ++y)
			{
				const float* s = ((float*) tmp[visualize_bg - 1].data) + y * sc->width;
				uchar* out = img.bits() + img.bytesPerLine() * y;
				for (int x = 0; x < sc->width; ++x)
				{
					if (*s > 0)
					{
						int v = *s * 0.5;
						if (v>255) v = 255;
						*out = 0; ++out;
						*out = (uchar) v; ++out; // positive is green
					}
					else
					{
						int v = -*s * 0.5;
						if (v>255) v = 255;
						*out = (uchar) v; ++out; // negative is red
						*out = 0; ++out;
					}
					++out;
					++s;
				}
			}

		} //if (visualize_bg > 0)
		frame_pixmap = QPixmap::fromImage(img, Qt::ColorOnly);
	}*/  //if (cooker && people_counter)

	new_frame_arrived = false;
}

void DetWindow::draw_logic_rect(const std::vector<GroundTruthPoint*> &vec,
	float margin, QPainter *painter, Qt::GlobalColor color0, Qt::GlobalColor color1)
{
	painter->save();
	for (std::vector<GroundTruthPoint*>::const_iterator it = vec.begin(); it != vec.end(); ++it)
	{
		const GroundTruthPoint *p = *it;
		Qt::GlobalColor color = color0;
		if (p->object_type != 0)
			color = color0;
		else
			color = color1;
		int x = p->center_x * rect().width() / 100;
		int y = p->center_y * rect().height() / 100;
		int w = p->width * rect().width() / 100;
		int h = p->height * rect().height() / 100;
		painter->setPen(QPen(color, 1));
		painter->setBrush(QBrush());
		painter->drawRect(x - w / 2, y - h / 2, w, h);
		QBrush br(color, Qt::Dense4Pattern);
		painter->setBrush(br);
		int border_w = int(w * margin);
		if (w > 5 && border_w > 2)
		{
			painter->drawRect(x - w / 2, y - h / 2, border_w, h);
			painter->drawRect(x + w / 2 - border_w, y - h / 2, border_w, h);
		}
		int border_h = int(h * margin);
		if (h > 5 && border_h > 2)
		{
			painter->drawRect(x - w / 2, y - h / 2, w, border_h);
			painter->drawRect(x - w / 2, y + h / 2 - border_h, w, border_h);
		}
		painter->drawText(x, y, QString::number(p->object_id));
	}
	painter->restore();
}

void DetWindow::draw_track(const GUITrack &track, QPainter *painter, bool active)
{
	QRect rect = get_rect_on_widget(track, cur_frame_num);
	if (rect.isEmpty())
		return;

	painter->save();
	QColor color;
	if (active)
		color = Qt::green;
	else
		color = Qt::blue;
	color.setAlpha(50);
	painter->setPen(QPen(color, 1));
	painter->setFont(QFont("Arial", 10));
	int x = rect.x();
	int y = rect.y();
	int w = rect.width();
	int h = rect.height();
	int border_w = std::min(w / 4, 30);
	int border_h = std::min(h / 4, 30);
	painter->drawLine(x + border_w, y, x + border_w, y + h);
	painter->drawLine(x + w - border_w, y, x + w - border_w, y+h);
	painter->drawLine(x, y + border_h, x + w, y + border_h);
	painter->drawLine(x, y + h - border_h, x + w, y + h - border_h);

	color.setAlpha(255);
	GUITrack::POINT_TYPE pt_type = track.point_type(cur_frame_num);
	af_assert(pt_type != GUITrack::POINT_ABSENT);  // already checked before
	if (pt_type == GUITrack::POINT_INTERPOLATED)
	{
		painter->setPen(QPen(color, 1, Qt::DashLine));
		painter->drawText(rect.center(), QString::number(track.object_id));
		if (active && draw_traces)
			draw_track_trace(track, painter, color);
	}
	else if (pt_type == GUITrack::POINT_EXTRAPOLATED)
	{
		painter->setPen(QPen(color, 1, Qt::DashLine));
	}
	else if (pt_type == GUITrack::POINT_BASE)
	{
		painter->setPen(QPen(color, 1, Qt::SolidLine));
		painter->drawText(rect.center(), QString::number(track.object_id));
		if (active && draw_traces)
			draw_track_trace(track, painter, color);
	}
	painter->drawRect(rect);

	painter->restore();
}

void DetWindow::draw_track_trace(const GUITrack &track,
	QPainter* painter, QColor color)
{
	if (cur_frame_num < track.start_frame || cur_frame_num > track.finish_frame)
		return;

	painter->save();

	int start = std::max(track.start_frame, cur_frame_num - TRACE_FRAMES_NUM);
	int finish = std::min(track.finish_frame, cur_frame_num + TRACE_FRAMES_NUM);

	for (int i = start; i <= finish; ++i)
	{
		QRect rect = get_rect_on_widget(track, i);
		if (track.point_type(i) == GUITrack::POINT_BASE)
			painter->setPen(QPen(color, 3));
		else
			painter->setPen(QPen(color, 1));
		painter->drawEllipse(rect.topLeft(), 1, 1);
		painter->drawEllipse(rect.bottomLeft() + QPoint(0, 1), 1, 1);
		painter->drawEllipse(rect.topRight() + QPoint(1, 0), 1, 1);
		painter->drawEllipse(rect.bottomRight() + QPoint(1, 1), 1, 1);
	}
	painter->restore();
}

void DetWindow::timeout()
{
	int skip_frames = 4;
	switch (movie_play_mode)
	{
	case PLAY:
		get_frame(1);
		visualize();
		update();
		break;
	case PLAY_FAST:
		while (skip_frames--)
			get_frame(1);
		visualize();
		update();
		break;
	default:

		break;
	}
}

void DetWindow::set_workmode(int mode)
{
	set_play_mode(PAUSE);
	active_zone = -1;
	workmode = mode;

	/*
	if (workmode & MODE_GROUND_PLANE)
	{
		if (people_counter && people_counter->detector)
			size_map = people_counter->detector->size_map;
	}
	else
		size_map_point = -1;
	*/
	picture_settings_update();
}

void DetWindow::detections_show(bool state)
{
	is_detections_visible = state;
	update();
}

void DetWindow::targets_show(bool state)
{
	is_targets_visible = state;
	update();
}

void DetWindow::results_show(bool state)
{
	is_results_visible = state;
	update();
}

void DetWindow::gt_show(bool state)
{
	is_gt_visible = state;
	update();
}

void DetWindow::zones_show(bool state)
{
	is_zones_visible = state;
	update();
}

void DetWindow::lost_objects_show(bool)
{
	picture_settings_update();
}

void DetWindow::feature_tracks_show(bool)
{
	picture_settings_update();
}

void DetWindow::lost_features_show(bool)
{
	picture_settings_update();
}

void DetWindow::prefilter_use(bool)
{
	picture_settings_update();
}

void DetWindow::bg_mask_show(bool)
{
	picture_settings_update();
}

std::string DetWindow::zone_selected(int zone_num)
{
	active_zone = zone_num;
	return zones[zone_num].type;
}

void DetWindow::zone_type_changed(int zone_num, const std::string &str)
{
	zones[zone_num].type = str;
}

void DetWindow::obj_params_to_ui()
{
	if (edited_track == -1)
		return;

	emit obj_size_changed(0, 0);
}

void DetWindow::ui_to_obj_params(double w, double h)
{
	update_result_point(edited_track, edited_type, -1, -1, float(w), float(h));

	visualize();
	update();
}

DemoWindow::DemoWindow()
	: ui(new Ui::DemoWindow), type_split_bit(8)
{
	ui->setupUi(this);
	mouse_label = new QLabel(this);
	track_label = new QLabel(this);
	frame_label = new QLabel(this);
	ui->statusbar->addWidget(mouse_label);
	ui->statusbar->addWidget(track_label);
	ui->statusbar->addPermanentWidget(frame_label);

	det_window = new DetWindow(this);
	ui->video_frame->layout()->addWidget(det_window);

	ui->mode->blockSignals(true);
	ui->mode->addItem(tr("Correction"), QVariant(DetWindow::MODE_CORRECT_CREATE));
	ui->mode->addItem(tr("Detect"), QVariant(DetWindow::MODE_DETECT));
	ui->mode->addItem(tr("Track"), QVariant(DetWindow::MODE_TRACK));
	ui->mode->addItem(tr("Full"), QVariant(DetWindow::MODE_DETECT | DetWindow::MODE_TRACK));
	ui->mode->addItem(tr("Zones"), QVariant(DetWindow::MODE_ZONE_CREATE));
	/*
	ui->mode->addItem(tr("Detection Area"), QVariant(DetWindow::MODE_CROP));
	ui->mode->addItem(tr("Classifier score"), QVariant(DetWindow::MODE_CLASSIFY));
	ui->mode->addItem(tr("Groud Plane"), QVariant(DetWindow::MODE_GROUND_PLANE));
	*/

	ui->mode->setCurrentIndex(0);
	ui->mode->blockSignals(false);

	ui->zone_type->blockSignals(true);
	ui->zone_number->blockSignals(true);
	ui->zone_type->addItem("none", QVariant(""));
	ui->zone_type->addItem("ignore", QVariant("ignore"));
	ui->zone_type->addItem("lookup", QVariant("lookup"));
	ui->zone_type->addItem("border", QVariant("border"));
	for (int i = 0; i < 15; ++i)
		ui->zone_number->addItem(QString::number(i), QVariant(i));
	ui->zone_type->blockSignals(false);
	ui->zone_number->blockSignals(false);

	//ui->object_type->blockSignals(true);
	//ui->object_type->blockSignals(false);

	ui->gt_obj_table->blockSignals(true);
	ui->result_obj_table->blockSignals(true);
	QStringList table_header;
	table_header << "obj ID";
	table_header << "points";
	table_header << "first fr";
	table_header << "last fr";
	table_header << "correctness";
	table_header << "flw";
	table_header << "type";
	ui->gt_obj_table->setColumnCount(table_header.size());
	ui->gt_obj_table->setHorizontalHeaderLabels(table_header);
	ui->gt_obj_table->resizeColumnsToContents();
	ui->result_obj_table->setColumnCount(table_header.size());
	ui->result_obj_table->setHorizontalHeaderLabels(table_header);
	ui->result_obj_table->resizeColumnsToContents();
	ui->gt_obj_table->blockSignals(false);
	ui->result_obj_table->blockSignals(false);

	connector();
}

DemoWindow::~DemoWindow()
{
	delete ui;
}

void DemoWindow::connector()
{
	connect(det_window, SIGNAL(mouse_label_changed(const QString &)),
		this, SLOT(set_mouse_label(const QString &)));
	connect(det_window, SIGNAL(frame_num_changed(const QString &)),
		this, SLOT(set_frame_label(const QString &)));
	connect(det_window, SIGNAL(obj_size_changed(int, int)),
		this, SLOT(set_object_size(int, int)));
	connect(det_window, SIGNAL(gt_track_changed(int)),
		this, SLOT(table_update(int)));
	connect(det_window, SIGNAL(gt_track_added(int)),
		this, SLOT(table_reload(int)));
	connect(det_window, SIGNAL(gt_track_removed(int, int)),
		this, SLOT(table_remove_row(int,int)));
	connect(det_window, SIGNAL(gt_track_selected()),
		this, SLOT(update_cur_type()));
}

void DemoWindow::reset_gt()
{
	results.in_video = task.is_movie();
	if (!task.correct_path.empty())
		on_load_gt_act_triggered();
	else
		log_warning("GT LOAD: incorrect ground truth path");
	det_window->results2tracks();

	objects_to_ui(det_window->gt_tracks, ui->gt_obj_table);
}

bool DemoWindow::table_track_update(const GUITrack &track, QTableWidget *ui_table)
{
	if (track.table_row == -1)
		return false;

	if (ui_table->rowCount() <= track.table_row)
		return false;

	ui_table->item(track.table_row, 0)->setText(QString::number(track.object_id));
	ui_table->item(track.table_row, 1)->setText(QString::number(track.points.size()));
	ui_table->item(track.table_row, 2)->setText(QString::number(track.start_frame));
	ui_table->item(track.table_row, 3)->setText(QString::number(track.finish_frame));
	QTableWidgetItem *item = ui_table->item(track.table_row, 4);
	item->setText(QString::number(track.found_percent));
	if (track.found_percent > 0.8f)
		item->setBackgroundColor(QColor(150, 255, 150));
	else if (track.found_percent > 0.3f)
		item->setBackgroundColor(QColor(255, 255, 150));
	else
		item->setBackgroundColor(QColor(255, 150, 150));
	ui_table->item(track.table_row, 5)->setText(QString::number(track.points_per_id.size()));
	ui_table->item(track.table_row, 6)->setText(QString::number(track.object_type));
	return true;
}

bool DemoWindow::table_track_remove(int row, QTableWidget *ui_table)
{
	if (row == -1)
		return false;

	if (ui_table->rowCount() <= row)
		return false;

	for (auto tr  = det_window->gt_tracks.begin();
			tr != det_window->gt_tracks.end(); ++tr)
	{
		GUITrack &track = tr->second;
		af_assert(track.table_row != row);
		if (track.table_row > row)
			--track.table_row;
	}
	ui_table->removeRow(row);
	return true;
}

bool DemoWindow::table_point_update(
	const GUITrackPoint & /* point */, QTableWidget * /* ui_table */)
{
	return false;
}

void DemoWindow::objects_to_ui(DetWindow::tracks_t &tracks, QTableWidget *ui_table)
{
	ui_table->clearContents();
	ui_table->model()->removeRows(0, ui_table->rowCount());
	for (auto it = tracks.begin(); it != tracks.end(); ++it)
	{
		GUITrack &trk = it->second;
		int cur_row = ui_table->rowCount();
		trk.table_row = cur_row;
		ui_table->insertRow(cur_row);
		QTableWidgetItem *item = new QTableWidgetItem(QString::number(it->first));
		ui_table->setItem(cur_row, 0, item);
		item = new QTableWidgetItem(QString::number(trk.points.size()));
		ui_table->setItem(cur_row, 1, item);
		item = new QTableWidgetItem(QString::number(trk.start_frame));
		ui_table->setItem(cur_row, 2, item);
		item = new QTableWidgetItem(QString::number(trk.finish_frame));
		ui_table->setItem(cur_row, 3, item);

		item = new QTableWidgetItem(QString::number(trk.found_percent));
		if (trk.found_percent > 0.8f)
			item->setBackgroundColor(QColor(150, 255, 150));
		else if (trk.found_percent > 0.3f)
			item->setBackgroundColor(QColor(255, 255, 150));
		else
			item->setBackgroundColor(QColor(255, 150, 150));
		ui_table->setItem(cur_row, 4, item);

		item = new QTableWidgetItem(QString::number(trk.points_per_id.size()));
		ui_table->setItem(cur_row, 5, item);
		item = new QTableWidgetItem(QString::number(trk.object_type));
		ui_table->setItem(cur_row, 6, item);
	}
}

void DemoWindow::objects_to_ui(PointList &data, QTableWidget *ui_table)
{
	ui_table->clearContents();
	ui_table->model()->removeRows(0, ui_table->rowCount());
	for (auto it = data.points.begin(); it != data.points.end(); ++it)
	{
		const GroundTruthPoint &pt = *it;
		int cur_row = ui_table->rowCount();
		ui_table->insertRow(cur_row);
		QTableWidgetItem *item = new QTableWidgetItem(QString::number(pt.object_id));
		ui_table->setItem(cur_row, 0, item);
		item = new QTableWidgetItem(QString::number(1));
		ui_table->setItem(cur_row, 1, item);
		item = new QTableWidgetItem(QString::number(pt.frame_num));
		ui_table->setItem(cur_row, 2, item);
		item = new QTableWidgetItem(QString::number(pt.frame_num));
		ui_table->setItem(cur_row, 3, item);
		item = new QTableWidgetItem(QString::number(pt.found_percent));
		if (pt.found_percent > 0.8f)
			item->setBackgroundColor(QColor(150, 255, 150));
		else if (pt.found_percent > 0.3f)
			item->setBackgroundColor(QColor(255, 255, 150));
		else
			item->setBackgroundColor(QColor(255, 150, 150));
		ui_table->setItem(cur_row, 4, item);

		item = new QTableWidgetItem(QString::number(pt.followers));
		ui_table->setItem(cur_row, 5, item);
		item = new QTableWidgetItem(QString::number(pt.object_type));
		ui_table->setItem(cur_row, 6, item);
	}
}

void DemoWindow::tables_reload()
{
	if (ui->table_track_view->isChecked())
	{
		objects_to_ui(det_window->gt_tracks, ui->gt_obj_table);
		objects_to_ui(det_window->found_tracks, ui->result_obj_table);
	}
	else
	{
		objects_to_ui(results.all_points_correct, ui->gt_obj_table);
		objects_to_ui(results.all_points_detected, ui->result_obj_table);
	}
}

void DemoWindow::keyPressEvent(QKeyEvent* ev)
{
	int key = ev->key();
	switch (key)
	{
	case Qt::Key_PageUp:
	case Qt::Key_Z:
		det_window->prev_frame();
		break;
	case Qt::Key_PageDown:
	case Qt::Key_X:
		det_window->next_frame();
		break;
	case Qt::Key_C:
		det_window->jump_frame();
		break;
	case Qt::Key_Space:
		det_window->set_play_mode(DetWindow::PLAY_PAUSE);
		break;
	case Qt::Key_Home:
		det_window->start_movie();
		break;
	case Qt::Key_End:
		det_window->end_movie();
		break;
	case Qt::Key_Insert:
	{
		int inttype = ui->object_type->currentData().toInt();
		int subtype = 0;
		if (ui->object_subtype->count())
			subtype = ui->object_subtype->currentData().toInt();
		det_window->add_sample_track(type_merge(inttype, subtype));
		break;
	}
	case Qt::Key_O:
		det_window->draw_traces = !det_window->draw_traces;
		break;
	case Qt::Key_F1:
		det_window->set_play_mode(DetWindow::PAUSE);
		on_help_act_triggered();
		break;
	case Qt::Key_F2:
		on_save_gt_act_triggered();
		break;
	case Qt::Key_F3:
		on_load_gt_act_triggered();
		break;
	case Qt::Key_Backspace:
		det_window->del_current_point();
		update();
		break;
	case Qt::Key_Delete:
		if (ev->modifiers() == Qt::ShiftModifier)
		{
			det_window->del_current_track();
		}
		else
		{
			det_window->del_current_point();
		}
		update();
		break;
	case Qt::Key_1:
		if (task.is_movie() && det_window->cur_sample_track)
		{
			det_window->cur_sample_track->extrapolation_enable =
				!det_window->cur_sample_track->extrapolation_enable;
			update();
		}
		break;
	case Qt::Key_2:
	case Qt::Key_3:
	case Qt::Key_4:
	case Qt::Key_5:
	case Qt::Key_6:
	case Qt::Key_7:
	case Qt::Key_8:
	case Qt::Key_9:
	case Qt::Key_0:
	case Qt::Key_Minus:
	case Qt::Key_Equal:
	{
		int k = key - Qt::Key_2;
		if (key==Qt::Key_0) k = 8;
		if (key==Qt::Key_Minus) k = 9;
		if (key==Qt::Key_Equal) k = 10;
		float mul = k * 2.0f;
		det_window->change_cur_point_size(mul, mul);
		break;
	}
	case Qt::Key_Up:
		if (ev->modifiers() == Qt::ShiftModifier)
			det_window->stretch_rect(0, 1);
		else
			det_window->move_rect(0, -1);
		break;
	case Qt::Key_Down:
		if (ev->modifiers() == Qt::ShiftModifier)
			det_window->stretch_rect(0, -1);
		else
			det_window->move_rect(0, 1);
		break;
	case Qt::Key_Left:
		if (ev->modifiers() == Qt::ShiftModifier)
			det_window->stretch_rect(-1, 0);
		else
			det_window->move_rect(-1, 0);
		break;
	case Qt::Key_Right:
		if (ev->modifiers() == Qt::ShiftModifier)
			det_window->stretch_rect(1, 0);
		else
			det_window->move_rect(1, 0);
		break;

	}
	//gt-editor

	ev->accept();
	det_window->kev(ev);
}

void DemoWindow::wheelEvent(QWheelEvent *ev)
{
	det_window->set_play_mode(DetWindow::PAUSE);
	int direction = ev->delta() > 0 ? 1 : -1;
	if (ev->orientation() == Qt::Horizontal) // or vertical+shift
	{
		if (det_window->cur_sample_track)
			det_window->stretch_rect(direction, 0);
	}
	else if (ev->modifiers() & Qt::ControlModifier)
	{
		if (det_window->cur_sample_track)
			det_window->stretch_rect(0, direction);
	}
	else
		det_window->get_frame(direction);

	/*int cur_mode = ui->mode->itemData(ui->mode->currentIndex()).toInt();
	//Qt::Horizontal is by default when alt is pressedQt::Horizontal
	if (cur_mode & DetWindow::MODE_CORRECT_CREATE)
	{
		if (ev->orientation() == Qt::Horizontal)
			ui->obj_size_w->setValue(ui->obj_size_w->value() + (ev->delta() > 0 ? 0.5 : -0.5));
		else
			ui->obj_size_h->setValue(ui->obj_size_h->value() + (ev->delta() > 0 ? 0.5 : -0.5));
	}
	else if (cur_mode & DetWindow::MODE_GROUND_PLANE)
	{
		bool horizontal = false;
		if (ev->orientation() == Qt::Horizontal) //or vertical+shift
			horizontal = true;

		bool minimal = false;
		if (ev->modifiers() & Qt::ControlModifier)
			minimal = true;

		double delta = ev->delta() > 0 ? 0.5 : -0.5;

		det_window->update_size_map_point(horizontal, minimal, delta);
	}
	*/
}

void DemoWindow::table_update(int track_id)
{
	log_state("Updating GUI tables");
	if (ui->table_track_view->isChecked())
	{
		auto it = det_window->gt_tracks.find(track_id);
		bool done = false;
		if (it != det_window->gt_tracks.end())
		{
			const GUITrack &t = it->second;
			done = table_track_update(t, ui->gt_obj_table);
		}
		if (!done)
			on_table_track_view_clicked();  // update all
	}
	else
		on_table_point_view_clicked();
}

void DemoWindow::table_remove_row(int /* track_id */, int row)
{
	log_state("Removing from GUI tables");
	if (ui->table_track_view->isChecked())
	{
		bool done = table_track_remove(row, ui->gt_obj_table);
		if (!done)
			on_table_track_view_clicked();  // update all
	}
	else
		on_table_point_view_clicked();
}

void DemoWindow::table_reload(int /* track_id */)
{
	log_state("Reloading GUI tables");
	if (ui->table_track_view->isChecked())
		on_table_track_view_clicked();
	else
		on_table_point_view_clicked();
}

void DemoWindow::update_cur_type()
{
	if (!det_window->cur_sample_track)
		return;

	int typenumber = det_window->cur_sample_track->object_type;

	int inttype, intsubtype;
	type_split(typenumber, &inttype, &intsubtype);
	int index = ui->object_type->findData(inttype);
	if (index == -1)
		 index = ui->object_type->findData(0);

	ui->object_type->blockSignals(true);
	ui->object_type->setCurrentIndex(index);
	ui->object_type->blockSignals(false);
	update_subtypes(intsubtype);
}

void DemoWindow::read_type_name()
{
	std::string undef = "undefined";
	type_name.insert(std::make_pair(0, undef));
	type_map_t::iterator itype =
			type_map.insert(std::make_pair(0, subtype_name_t())).first;

	std::ifstream ifs;
	ifs.open(task.obj_type_legend, std::ifstream::in);

	if (ifs.is_open())
	{
		std::string str;
		while (ifs.good())
		{
			std::getline(ifs, str);
			std::list<std::string> elems = aifil::split(str, '\t');
			if (elems.size() % 2 != 0 || elems.size() < 2)
			{
				aifil::log_warning("bad line in types");
				continue;
			}
			std::list<std::string>::iterator it = elems.begin();
			int type = std::stoi(*it, nullptr, 10);
			++it;
			std::string name = *it;
			++it;
			type_name.insert(std::make_pair(type, name));
			type_map_t::iterator itype = type_map.insert(std::make_pair(type, subtype_name_t())).first;

			aifil::log_warning("type = %d, name = %s", type, name.c_str());

			while (it != elems.end())
			{
				int subtype = std::stoi(*it, nullptr, 10);
				++it;
				name = *it;
				++it;
				itype->second.insert(std::make_pair(subtype, name));
				aifil::log_warning("subtype = %d, name = %s", subtype, name.c_str());
			}
		}

		ifs.close();
	}

	ui->object_type->blockSignals(true);
	if (type_name.size() == 1)
	{
		ui->object_type->addItem(undef.c_str(), QVariant(0));
		for (int i = 1; i <= 250; ++i)
			ui->object_type->addItem(QString::number(i), QVariant(i));
	}
	else
	{
		for(type_name_t::iterator it = type_name.begin(); it != type_name.end(); ++it)
			ui->object_type->addItem(it->second.c_str(), QVariant(it->first));
	}
	int curindex = ui->object_type->findText(undef.c_str());
	ui->object_type->setCurrentIndex(curindex);
	ui->object_type->blockSignals(false);
}

void DemoWindow::update_subtypes(int intsubtype)
{
	ui->object_subtype->blockSignals(true);
	ui->object_subtype->clear();

	int inttype = ui->object_type->currentData().toInt();
	auto itype = type_map.find(inttype);
	if (itype == type_map.end() || itype->second.empty())
		return;

	std::string undef = "undefined";
	ui->object_subtype->addItem(undef.c_str(), QVariant(0));
	for (auto it = itype->second.begin(); it != itype->second.end(); ++it)
		ui->object_subtype->addItem(it->second.c_str(), QVariant(it->first));

	int index = ui->object_subtype->findData(intsubtype);
	ui->object_subtype->setCurrentIndex(index);
	ui->object_subtype->blockSignals(false);
}

int DemoWindow::type_number(const std::string &type)
{
	int inttype = 0;
	for (auto it = type_name.begin(); it != type_name.end(); ++it)
	{
		if (it->second == type)
		{
			inttype = it->first;
			break;
		}
	}
	return inttype;
}

int DemoWindow::subtype_number(int inttype, const std::string &subtype)
{
	int intsubtype = 0;
	if (inttype == 0)
		return intsubtype;

	type_map_t::iterator itype = type_map.find(inttype);
	for (subtype_name_t::iterator it = itype->second.begin(); it != itype->second.end(); ++it)
	{
		if (it->second == subtype)
		{
			intsubtype = it->first;
			break;
		}
	}

	return intsubtype;
}


void DemoWindow::type_string(int typenumber, std::string &type, std::string &subtype)
{
	int inttype, intsubtype;
	type_split(typenumber, &inttype, &intsubtype);

	type_name_t::iterator itype = type_name.find(inttype);
	if (itype == type_name.end())
	{
		type = std::to_string(inttype);
		subtype = std::to_string(intsubtype);
		return;
	}
	type = itype->second;

	type_map_t::iterator imtype = type_map.find(inttype);
	subtype_name_t::iterator isubtype = imtype->second.find(intsubtype);

	if (isubtype == imtype->second.end())
	{
		if (intsubtype == 0)
			subtype = "undefined";
		else
			subtype = std::to_string(intsubtype);
		return;
	}
	subtype = isubtype->second;
}

void DemoWindow::type_split(int typenumber, int *type, int *subtype)
{
	(*subtype) = typenumber >> type_split_bit;
	(*type) = typenumber - ((*subtype) << type_split_bit);
}

int DemoWindow::type_merge(int type, int subtype)
{
	return (subtype << type_split_bit) + type;
}


void DemoWindow::renew_detector_stat()
{

}

void DemoWindow::set_mouse_label(const QString &text)
{
	mouse_label->setText(text);
}

void DemoWindow::set_frame_label(const QString &text)
{
	frame_label->setText(text);
}

void DemoWindow::set_object_size(int w, int h)
{
}

void DemoWindow::on_mode_currentIndexChanged(int index)
{
	int workmode = ui->mode->itemData(index).toInt();
	det_window->set_workmode(workmode);
}

void DemoWindow::on_play_button_clicked()
{
	det_window->set_play_mode(DetWindow::PLAY_PAUSE);
}

void DemoWindow::on_prev_button_clicked()
{
	det_window->prev_frame();
}

void DemoWindow::on_next_button_clicked()
{
	det_window->next_frame();
}

void DemoWindow::on_jump_button_clicked()
{
	det_window->jump_frame();
	det_window->update();
}

void DemoWindow::on_fast_button_clicked()
{
	det_window->set_play_mode(DetWindow::PLAY_FAST);
}

void DemoWindow::on_zone_number_currentIndexChanged(int index)
{
	ui->zone_type->blockSignals(true);
	ui->zone_type->setCurrentIndex(
		ui->zone_type->findData(
			det_window->zone_selected(
				ui->zone_number->itemData(index).toInt()
				).c_str()));
	ui->zone_type->blockSignals(false);
}

void DemoWindow::on_zone_type_currentIndexChanged(int index)
{
	det_window->zone_type_changed(
		ui->zone_number->itemData(ui->zone_number->currentIndex()).toInt(),
		ui->zone_type->itemData(index).toString().toLatin1().data());
}

void DemoWindow::on_gt_obj_table_clicked(const QModelIndex &index)
{
	int obj_id = ui->gt_obj_table->item(index.row(), 0)->text().toInt();
	det_window->cur_sample_track = &det_window->gt_tracks.find(obj_id)->second;
	int first_frame, last_frame;
	if (ui->table_track_view->isChecked())
	{
		first_frame = det_window->cur_sample_track->start_frame;
		last_frame = det_window->cur_sample_track->finish_frame;
	}
	else
	{
		first_frame = ui->gt_obj_table->item(index.row(), 2)->text().toInt();
		last_frame = ui->gt_obj_table->item(index.row(), 3)->text().toInt();
	}
	update_cur_type();
	det_window->edited_track = obj_id;
	det_window->ensure_visible(first_frame, last_frame);
	det_window->update();
	aifil::log_state("GT: now editing track '%d'", det_window->edited_track);
}

void DemoWindow::on_result_obj_table_clicked(const QModelIndex &index)
{
	if (ui->table_track_view->isChecked())
	{
		int obj_id = ui->result_obj_table->item(index.row(), 0)->text().toInt();
		det_window->cur_found_track = &det_window->found_tracks.find(obj_id)->second;
		int first_frame, last_frame;
		if (ui->table_track_view->isChecked())
		{
			first_frame = det_window->cur_found_track->start_frame;
			last_frame = det_window->cur_found_track->finish_frame;
		}
		else
		{
			first_frame = ui->gt_obj_table->item(index.row(), 2)->text().toInt();
			last_frame = ui->gt_obj_table->item(index.row(), 3)->text().toInt();
		}
		update_cur_type();
		det_window->ensure_visible(first_frame, last_frame);
		det_window->update();
	}
}

void DemoWindow::on_table_track_view_clicked()
{
	// log_state("DemoWindow::on_table_track_view_clicked");
	tables_reload();
}

void DemoWindow::on_table_point_view_clicked()
{
	// log_state("DemoWindow::on_table_point_view_clicked");
	tables_reload();
}
/*
void DemoWindow::on_object_type_editingFinished()
{
	if (det_window->cur_obj_track_sample)
		det_window->cur_obj_track_sample->update_type(
			ui->object_type->text().toInt(), &results.all_points_correct);
}
*/
void DemoWindow::on_object_type_currentIndexChanged(const QString &)
{
	if (!det_window->cur_sample_track)
		return;

	int inttype = ui->object_type->currentData().toInt();
	det_window->cur_sample_track->update_type(
			type_merge(inttype, 0), &results.all_points_correct);
	update_subtypes(0);
}

void DemoWindow::on_object_subtype_currentIndexChanged(const QString &)
{
	if (!det_window->cur_sample_track)
		return;

	int inttype = ui->object_type->currentData().toInt();
	int intsubtype = 0;
	if (ui->object_subtype->count())
		intsubtype = ui->object_subtype->currentData().toInt();
	det_window->cur_sample_track->update_type(
			type_merge(inttype, intsubtype), &results.all_points_correct);
}

void DemoWindow::on_obj_size_w_valueChanged(double w)
{
	det_window->ui_to_obj_params(w, ui->obj_size_h->value());
}

void DemoWindow::on_obj_size_h_valueChanged(double h)
{
	det_window->ui_to_obj_params(ui->obj_size_w->value(), h);
}

void DemoWindow::on_open_task_act_triggered()
{
	det_window->set_play_mode(DetWindow::PAUSE);
	QString task_file = QFileDialog::getOpenFileName(
		this, tr("Open Task File"),
		"",
		tr("All Files (*)"));
	read_new_task(task_file.toUtf8().data());
}

void DemoWindow::read_new_task(const std::string &task_file_name)
{
	try {
		task.prepare_to_work(task_file_name);
	} catch (const std::runtime_error &e) {
		log_error("Wrong task '%s': %s", task_file_name.c_str(), e.what());
		return;
	}
	reset_gt();
	if (!task.result_path.empty())
		on_load_result_act_triggered();
	else
		log_warning("GT LOAD: no result path");
	det_window->reset_all();
	read_type_name();
}

void DemoWindow::on_load_gt_act_triggered()
{
	results.read_points(task.correct_path, true, true);
	det_window->convert_name2int();
	DetectorStat stat;
	results.collect_detector_stat(task.object_type, stat);
	//results.collect_tracker_stat();
	tables_reload();
}

void DemoWindow::on_load_result_act_triggered()
{
	results.read_points(task.result_path, false, true);
	//det_window->convert_name2int();
	DetectorStat stat;
	results.collect_detector_stat(task.object_type, stat);
	//results.collect_tracker_stat();
	tables_reload();
}

void DemoWindow::on_save_gt_act_triggered()
{
	results.all_points_correct.write(task.correct_path, true);
}

void DemoWindow::on_save_result_act_triggered()
{
	results.all_points_detected.write(task.result_path);
}

void DemoWindow::on_gt_checkbox_toggled(bool checked)
{
	det_window->gt_show(checked);
}

void DemoWindow::on_results_checkbox_toggled(bool checked)
{
	det_window->results_show(checked);
}

void DemoWindow::on_detections_checkbox_toggled(bool checked)
{
	det_window->detections_show(checked);
}

void DemoWindow::on_targets_checkbox_toggled(bool checked)
{
	det_window->targets_show(checked);
}

void DemoWindow::on_zones_checkbox_toggled(bool checked)
{
	det_window->zones_show(checked);
}

void DemoWindow::on_exit_act_triggered()
{
	close();
}

void DemoWindow::on_help_act_triggered()
{
	QString str;
	str += "<b>GENERAL COMMANDS</b><br>";
	str += "<b>F1</b>: help<br>";
	str += "<b>F2</b>: save<br>";
	str += "<b>F3</b>: load<br><br>";

	str += "<b>MOVIE NAVIGATION</b><br>";
	str += "<b>Z</b> or <b>PgUp</b>: prev frame<br>";
	str += "<b>X</b> or <b>PgDown</b>: prev frame<br>";
	str += QString("<b>C</b>: %1 frames forward<br>").arg(DetWindow::JUMP_FRAMES);
	str += "<b>space</b>: play/stop<br>";
	str += "<b>home</b>: go to the first frame<br>";
	str += "<b>end</b>: go to the last frame<br><br>";

	str += "<b>TRACKS MANIPULATION</b><br>";
	str += "<b>insert</b>: create track<br>";
	str += "<b>ctrl</b>+mouse click on point: clone point to new track<br>";
	str += "<b>del</b> or <b>backspace</b>: delete point<br>";
	str += "<b>shift</b>+<b>del</b>: delete track<br>";
	str += "<b>1</b>: extrapolation enable/disable<br>";
	str += "from '<b>2</b>' to '<b>=</b>': set object size<br>";
	str += "<b>arrow</b>: move object<br>";
	str += "<b>shift</b>+<b>arrow</b>: resize object<br><br>";

	QMessageBox *box = new QMessageBox(QMessageBox::Information,
		"Help", str, QMessageBox::Ok, this,  Qt::Dialog | Qt::MSWindowsFixedSizeDialogHint);
	box->show();
}

void DemoWindow::on_about_act_triggered()
{
}

void DemoWindow::on_compute_stat_act_triggered()
{
	results.save_detector_stat(task.result_splitext_0() + ".det_stat", task.object_type);
}

}  // namespace ground_truth

int main(int argc, char *argv[])
{
	using namespace anfisa;
	using namespace ground_truth;

	QApplication app(argc, argv);
	try
	{
		DemoWindow demo;
		if (argc == 2)
			demo.read_new_task(argv[1]);
		demo.show();

		app.exec();
	}
	catch (const std::exception& e)
	{
		aifil::log_error("ERROR: %s\n", e.what());
	}
	printf("\n");

	return 0;
}
