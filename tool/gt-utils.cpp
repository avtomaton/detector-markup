#include "gt-utils.h"

#include <core/io-structures.h>

#include <common/math-helpers.h>
#include <common/errutils.h>
#include <common/fileutils.h>
#include <common/logging.h>
#include <common/profiler.h>
#include <common/stringutils.h>
#include <common/timeutils.h>

#include <iterator>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <boost/filesystem.hpp>

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

using aifil::log_state;
using aifil::rect_similarity;

namespace ground_truth {

bool Results::in_video = true;

Results::Results()
	: last_detection(0)
{
	det_metric_same = 0.5f;
	det_metric_good = 0.2f;
	det_metric_bad = FLT_EPSILON;
	det_followers_th = det_metric_bad;
}

static float gt_point_similarity(const GroundTruthPoint &r1, const GroundTruthPoint &r2)
{
	cv::Rect_<float> rect1(r1.center_x, r1.center_y, r1.width, r1.height);
	cv::Rect_<float> rect2(r2.center_x, r2.center_y, r2.width, r2.height);
	return rect_similarity(rect1, rect2);
}

int hg_revision()
{
	int rev = 0;
	FILE *tmp = popen("hg summary", "r");
	if (!tmp)
		return rev;

	char buf0[1024];
	char buf1[1024];
	if (fscanf(tmp, "%s %d:%s", buf0, &rev, buf1) != 3)
		aifil::log_warning("can't get HG revision info");
	pclose(tmp);
	return rev;
}

void Task::defaults()
{
	result_path = "result/";
	export_path = "export/";
	dump_path = "dumps/";

	resize_w = 0;
	resize_h = 0;
	wanted_fps = 25;
	skip_frames = 0;
	detector_skip_frames = 0;
	frames_cache_size = 0;
	stop_frame = -1;

	object_type = 0;

	dump_images = 0;
	dump_pedantic_size = 0;
	dump_type = "icf_dssl";

	interpolation = "linear";
	interpolation_points = 4;
}

Task::Task()
{
	strings["action"] = &action;

	strings["movie-path"] = &movie_path;
	strings["photo-dir"] = &photo_dir;
	strings["obj-type-legend"] = &obj_type_legend;

	strings["worker-type"] = &worker_type;
	ints["object-type"] = &object_type;

	strings["correct-path"] = &correct_path;
	strings["dump-path"] = &dump_path;
	strings["export-path"] = &export_path;
	strings["result-path"] = &result_path;
	strings["settings-path"] = &correct_path;

	strings["dump-type"] = &dump_type;
	strings["dump-prefix"] = &dump_prefix;
	ints["dump-images"] = &dump_images;
	ints["dump-pedantic-size"] = &dump_pedantic_size;
	callbacks["dump-tile-params"] = dump_tile_parse;

	strings["colorspace"] = &colorspace;

	ints["wanted-fps"] = &wanted_fps;
	ints["frames-cahche-size"] = &frames_cache_size;
	ints["skip-frames"] = &skip_frames;
	ints["detector-skip-frames"] = &detector_skip_frames;
	ints["stop-frame"] = &stop_frame;

	callbacks["resize"] = resize_parse;
	callbacks["erase"] = erase_file;

	callbacks["ground-plane"] = ground_plane_parse;
	strings["border-coords"] = &border_coords;

	strings["interpolation"] = &interpolation;
	ints["interpolation-points"] = &interpolation_points;

	defaults();
}

std::string Task::correct_splitext_0() const
{
	return std::get<0>(aifil::splitext(correct_path));
}

std::string Task::result_splitext_0() const
{
	return std::get<0>(aifil::splitext(result_path));
}

// if paths are folders (not files for reading/writing),
// append task name and corresponding extension to paths
void Task::cook()
{
	// TODO: check other possible extensions
	if (!aifil::endswith(correct_path, gt_base_ext))
		correct_path = correct_path + "/" + my_name + gt_base_ext;
	if (!aifil::endswith(result_path, gt_det_points_ext))
		result_path = result_path + "/" + my_name + gt_det_points_ext;
}

bool Task::is_movie()
{
	return !movie_path.empty();
}

void Task::resize_parse(ConfigParser* myself, const std::string &val)
{
	Task *self = static_cast<Task*>(myself);
	self->resize_w = 0;
	self->resize_h = 0;

	if (sscanf(val.c_str(), "%d %d", &self->resize_w, &self->resize_h) != 2)
		throw std::runtime_error("invalid resize parameters");
}

void Task::erase_file(ConfigParser*, const std::string &val)
{
	if (remove(val.c_str()) == 0)
		return;

	std::string cmd = "rm -f " + val;
	aifil::log_state("cannot erase '%s': %s\nmaybe pattern, trying '%s'...",
		val.c_str(), strerror(errno), cmd.c_str());

	if (system(cmd.c_str()))
		aifil::log_state("cannot erase '%s': %s", val.c_str(), strerror(errno));
}

void Task::ground_plane_parse(ConfigParser* myself, const std::string &val)
{
	Task *self = static_cast<Task*>(myself);
	self->ground_plane = val;
	int pos = 0;
	while ((pos = self->ground_plane.find_first_of(",")) > 0)
		self->ground_plane.replace(pos, 1, 1, '\n');
}

void Task::dump_tile_parse(ConfigParser* myself, const std::string &val)
{
	DumpStatICF rec;
	int r = sscanf(val.c_str(), "%d %d %d %d %d %d",
		&rec.obj_w, &rec.obj_h,
		&rec.margin_top, &rec.margin_right, &rec.margin_bottom, &rec.margin_left);
	if (r != 6)
		throw std::runtime_error("dump-icf: need 6 integers");

	rec.tile_w = rec.obj_w + rec.margin_left + rec.margin_right;
	rec.tile_h = rec.obj_h + rec.margin_top + rec.margin_bottom;
	Task* self = static_cast<Task*>(myself);
	self->dump_tile_params.push_back(rec);
}

void Task::prepare_to_work(const std::string &task_path)
{
	read(task_path);
	cook();
}

GroundTruthPoint::GroundTruthPoint()
	: frame_num(0),
	center_x(50.0f), center_y(50.0f), width(10.0f), height(10.0f),
	object_id(0), object_type(0), is_base(true), skipped(false),
	found(false), followers(0), found_percent(0)
{
}

GroundTruthPoint::GroundTruthPoint(float cx, float cy, float w, float h)
	: frame_num(0),
	center_x(cx), center_y(cy), width(w), height(h),
	object_id(0), object_type(0), is_base(true), skipped(false),
	found(false), followers(0), found_percent(0)
{
}

bool GroundTruthPoint::empty() const
{
	return pic_name.empty() && frame_num == 0;
}

bool GroundTruthPoint::operator <(const GroundTruthPoint &other) const
{
	bool equal = false;
	if (frame_num)
	{
		if (frame_num < other.frame_num)
			return true;
		else if (frame_num == other.frame_num)
			equal = true;
	}
	if (!pic_name.empty())
	{
		if (pic_name < other.pic_name)
			return true;
		else if (pic_name == other.pic_name)
			equal = true;
	}

	if (equal && object_id < other.object_id)
		return true;

	return false;
}

void GroundTruthPoint::set_coords(float new_x, float new_y, float new_w, float new_h)
{
	if (new_x == -1)
		new_x = center_x;
	if (new_y == -1)
		new_y = center_y;
	if (new_w == -1)
		new_w = width;
	if (new_h == -1)
		new_h = height;

	float half_w = new_w / 2.0f;
	float half_h = new_h / 2.0f;

	float left = std::max(0.0f, new_x - half_w);
	float right = std::min(100.0f, new_x + half_w);
	float up = std::max(0.0f, new_y - half_h);
	float down = std::min(100.0f, new_y + half_h);
	width = right - left;
	height = down - up;
	center_x = (right + left) / 2.0f;
	center_y = (up + down) / 2.0f;
}

cv::Rect GroundTruthPoint::rect(int frame_w, int frame_h) const
{
	return cv::Rect(int(frame_w * (center_x - width / 2) / 100.0f + 0.5f),
		int(frame_h * (center_y - height / 2) / 100.0f + 0.5f),
		int(frame_w * width / 100.0f + 0.5f),
		int(frame_h * height / 100.0f + 0.5f));
}

void GroundTruthPoint::set_rect(const cv::Rect &rect, int frame_w, int frame_h)
{
	center_x = (rect.x + rect.width / 2) * 100.0f / frame_w;
	center_y = (rect.y + rect.height / 2) * 100.0f / frame_h;
	width = rect.width * 100.0f / frame_w;
	height = rect.height * 100.0f / frame_h;
}

void GroundTruthPoint::clear_stat()
{
	found = false;
	followers = 0;
	found_percent = 0;
}

GroundTruthPoint GroundTruthPoint::interp_linear(
	const GroundTruthPoint &ob0, const GroundTruthPoint &ob1, int frame_cur)
{
	if (frame_cur == ob0.frame_num)
		return ob0;
	else if (frame_cur == ob1.frame_num)
		return ob1;

	GroundTruthPoint result;
	float d0_div_d = float(frame_cur - ob0.frame_num) / float(ob1.frame_num - ob0.frame_num);
	result.center_x = (ob1.center_x - ob0.center_x) * d0_div_d + ob0.center_x;
	result.center_y = (ob1.center_y - ob0.center_y) * d0_div_d + ob0.center_y;
	result.width = (ob1.width - ob0.width) * d0_div_d + ob0.width;
	result.height = (ob1.height - ob0.height) * d0_div_d + ob0.height;
	result.frame_num = frame_cur;
	result.object_id = ob0.object_id;
	result.object_type = ob0.object_type;
	result.is_base = false;

	return result;
}

GroundTruthPoint GroundTruthPoint::interp_polynom(
	const std::vector<GroundTruthPoint> &obs, int frame_cur, int count_points)
{
	for (auto it = obs.begin(); it != obs.end(); ++it)
		if (it->frame_num == frame_cur)
			return *it;

	GroundTruthPoint result;
	result.frame_num = frame_cur;
	result.object_id = obs[0].object_id;
	result.object_type = obs[0].object_type;
	result.is_base = false;

	auto low = std::lower_bound(obs.begin(), obs.end(), frame_cur,
		[](const GroundTruthPoint &lhs, int val)->bool
	{
		return lhs.frame_num < val;
	});

	int obssize = obs.size();

	int i0, i1;
	if (count_points >= obssize)
	{
		i0 = 0;
		i1 = obssize;
	}
	else
	{
		i0 = std::distance(obs.begin(), low) - count_points / 2;
		i1 = i0 + count_points;
		if (i0 < 0)
		{
			i1 -= i0;
			i0 = 0;
		}
		else if (i1 > obssize)
		{
			i0 -= i1 - obssize;
			i1 = obssize;
		}
	}

	int N = i1 - i0;

	std::vector<GroundTruthPoint> p(N);
	for (int i = i0; i < i1; ++i)
		p[i - i0] = obs[i];

	for (int j = 1; j < N; ++j)
	{
		for (int i = 0; i < N - j; ++i)
		{
			float diff0 = float(p[i + j].frame_num - frame_cur);
			float diff1 = float(frame_cur - p[i].frame_num);
			float diff2 = float(p[i + j].frame_num - p[i].frame_num);
			p[i].center_x = (p[i].center_x * diff0 + p[i + 1].center_x * diff1) / diff2;
			p[i].center_y = (p[i].center_y * diff0 + p[i + 1].center_y * diff1) / diff2;
			p[i].width = (p[i].width * diff0 + p[i + 1].width * diff1) / diff2;
			p[i].height = (p[i].height * diff0 + p[i + 1].height * diff1) / diff2;
		}
	}

	result.center_x = p[0].center_x;
	result.center_y = p[0].center_y;
	result.width = p[0].width;
	result.height = p[0].height;

	return result;
}

ObjectTrack::ObjectTrack(const GroundTruthPoint &point, INTERPOLATION interp):
	interpolation(interp), interpolation_points(4),
	object_id(point.object_id), object_type(point.object_type),
	extrapolation_enable(false),
	start_frame(0), finish_frame(0)
{
	//don't interpolate just one point
	update_base_pt(point, false);
}

ObjectTrack::POINT_TYPE ObjectTrack::point_type(int frame) const
{
	auto it = points.find(frame);
	if (it != points.end() && it->second.is_base)
		return POINT_BASE;
	else if (it != points.end() && !it->second.is_base)
		return POINT_INTERPOLATED;
	else if (extrapolation_enable &&
			(frame < start_frame || frame > finish_frame))
		return POINT_EXTRAPOLATED;
	else
		return POINT_ABSENT;
}

void ObjectTrack::update(bool interpolate)
{
	if (points.empty())
		return;

	// update borders
	start_frame = INT_MAX;
	finish_frame = 0;
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		if (!it->second.is_base)
			continue;

		if (start_frame == INT_MAX)
			start_frame = it->second.frame_num;
		finish_frame = it->second.frame_num;
	}

	if (!interpolate)
		return;

	points_t new_points;
	if (interpolation == INTERP_POLYNOM)
	{
		std::vector<GroundTruthPoint> keys;
		for (auto it = points.begin(); it != points.end(); ++it)
		{
			if (!it->second.is_base)
				continue;

			new_points[it->first] = it->second;  // add base point
			keys.push_back(it->second);
		}
		af_assert(keys.size());  // must be at least one base point
		for (int i = start_frame + 1; i < finish_frame; ++i)
		{
			if (new_points.find(i) == new_points.end())
				new_points[i] = GroundTruthPoint::interp_polynom(keys, i, interpolation_points);
		}
	}
	else  // linear
	{
		auto left = points.end();
		auto right = points.end();
		for (auto it = points.begin(); it != points.end(); ++it)
		{
			if (!it->second.is_base)
				continue;

			new_points[it->first] = it->second;  // add base point
			if (left == points.end())
				left = it;
			if (right == points.end() || left == right)
				right = it;

			// update points between 2 base points
			if (left != right)
			{
				for (int i = left->first + 1; i < right->first; ++i)
					new_points[i] = GroundTruthPoint::interp_linear(
						left->second, right->second, i);
				left = right;
			}
		}
	}
	std::swap(points, new_points);
}

void ObjectTrack::update_type(int new_type, PointList *pt_list)
{
	object_type = new_type;
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		it->second.object_type = new_type;
		if (pt_list)
			pt_list->update_point(it->second);
	}
}

GroundTruthPoint ObjectTrack::extrapolate(int frame) const
{
	GroundTruthPoint result;
	af_assert(frame < start_frame || frame > finish_frame);
	af_assert(extrapolation_enable);

	// track is one point - no any interpolation
	// copy point and assign proper frame number
	if (points.size() == 1)
	{
		result = points.begin()->second;
		result.frame_num = frame;
		af_assert(result.is_base);
		return result;
	}

	if (interpolation == INTERP_POLYNOM)
	{
		std::vector<GroundTruthPoint> keys;
		for (auto it = points.begin(); it != points.end(); ++it)
		{
			if (it->second.is_base)
				keys.push_back(it->second);
		}
		result = GroundTruthPoint::interp_polynom(keys, frame, interpolation_points);
	}
	else  // linear
	{
		// find 2 points near required border (left or right)
		// and interpolate point according to these 2 points
		if (frame < start_frame)
		{
			auto it0 = points.begin();
			af_assert(it0->second.is_base);  // first point must be base

			auto it1 = it0;
			do
				++it1;
			while (it1 != points.end() && !it1->second.is_base);
			result = GroundTruthPoint::interp_linear(
				it0->second, it1->second, frame);
		}
		else  // after last point
		{
			auto it0 = points.rbegin();
			af_assert(it0->second.is_base);  // last point must be base

			auto it1 = it0;
			do
				++it1;
			while (it1 != points.rend() && !it1->second.is_base);
			result = GroundTruthPoint::interp_linear(
				it0->second, it1->second, frame);
		}
	}
	return result;
}

GroundTruthPoint ObjectTrack::get_point(int frame) const
{
	if (start_frame <= frame && frame <= finish_frame)
	{
		points_t::const_iterator it = points.find(frame);
		if (it != points.end())
			return it->second;
	}
	else if (extrapolation_enable)
		return extrapolate(frame);
	return GroundTruthPoint();
}

void ObjectTrack::update_base_pt(const GroundTruthPoint &ob, bool interpolate)
{
	points[ob.frame_num] = ob;
	points[ob.frame_num].is_base = true;
	if (start_frame == 0 || start_frame > ob.frame_num)
		start_frame = ob.frame_num;
	if (finish_frame < ob.frame_num)
		finish_frame = ob.frame_num;

	// TODO: update only part of the track
	update(interpolate);
}

void ObjectTrack::find_near_frames(int frame, int& frame_low, int& frame_high) const
{
	frame_high = -1;
	frame_low = -1;

	if (frame < start_frame)
	{
		frame_low = -1;
		frame_high = start_frame;
		return;
	}
	if (frame > finish_frame)
	{
		frame_low = finish_frame;
		frame_high = -1;
		return;
	}

	auto left = points.begin();
	auto right = points.begin();
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		if (!it->second.is_base)
			continue;

		if (frame == it->second.frame_num)
		{
			frame_high = it->second.frame_num;
			frame_low = it->second.frame_num;
			return;
		}

		if (frame_low == -1)
		{
			frame_low = it->second.frame_num;
			left = it;
		}
		if (frame_high == -1 || left == right)
		{
			frame_high = it->second.frame_num;
			right = it;
		}
		if (left != right)
		{
			if (frame_low < frame && frame < frame_high)
				return;
			left = right;
		}
	}
}

int ObjectTrack::points_base() const
{
	int count = 0;
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		if (it->second.is_base)
			++count;
	}
	return count;
}

int ObjectTrack::points_found() const
{
	int count = 0;
	for (auto it = points.begin(); it != points.end(); ++it)
	{
		if (it->second.found)
			++count;
	}
	return count;
}

PointList::PointList()
	: found_percent(0)
{
	setlocale(LC_NUMERIC, "C");
}

void PointList::seek()
{
	if (Results::in_video)
		seek(-1);
	else
		seek("");
}

void PointList::seek(int frame_num)
{
	//af_assert(Results::in_video && "seeking in files dir by frame number is restricted");
	if (points.empty())
		return;

	if (frame_num == -1)
		current_pos = points.begin();
	else
	{
		while (current_pos->frame_num >= frame_num && current_pos != points.begin())
			--current_pos;
		while (current_pos->frame_num < frame_num && current_pos != points.end())
			++current_pos;
		if (current_pos == points.end())
			--current_pos;
	}
}

void PointList::seek(const std::string &frame_name)
{
	//af_assert(!Results::in_video && "seeking in video by frame name is restricted");

	if (points.empty())
		return;

	current_pos = points.begin();
	while (current_pos != points.end() && current_pos->pic_name != frame_name)
		++current_pos;
	if (current_pos == points.end())  // no such name: return to beginning
		current_pos = points.begin();
}

PointList::points_t::iterator PointList::seek_first_larger(int frame_num)
{
	points_t::iterator it = points.end();

	if (points.empty())
		return it;

	for (it = current_pos; it != points.end() && it->frame_num <= frame_num; ++it)
		;

	return it;
}

PointList::points_t::iterator PointList::seek_first_larger(const std::string &frame_name)
{
	points_t::iterator it = points.end();
	if (points.empty())
		return it;

	seek(frame_name);
	for (it = current_pos; it != points.end() && it->pic_name == frame_name; ++it)
		;
	return it;
}

std::vector<GroundTruthPoint*> PointList::select_by_frame(int frame_num)
{
	seek(frame_num);

	std::vector<GroundTruthPoint*> vec;
	if (points.empty())
		return vec;

	points_t::iterator it = current_pos;
	for ( ; it != points.end() && it->frame_num == frame_num; ++it)
		vec.push_back(&(*it));

	return vec;
}

std::vector<GroundTruthPoint*> PointList::select_by_frame(const std::string &frame_name)
{
	seek(frame_name);

	std::vector<GroundTruthPoint*> vec;
	if (points.empty())
		return vec;

	points_t::iterator it = current_pos;
	for ( ; it != points.end() && it->pic_name == frame_name; ++it)
		vec.push_back(&(*it));

	return vec;
}

int PointList::points_found() const
{
	int count = 0;
	for (points_t::const_iterator it = points.begin(); it != points.end(); ++it)
	{
		if (it->found)
			++count;
	}
	return count;
}

int PointList::points_not_skipped() const
{
	int count = 0;
	for (points_t::const_iterator it = points.begin(); it != points.end(); ++it)
	{
		if (!it->skipped)
			++count;
	}
	return count;
}

int PointList::first_frame() const
{
	if (!Results::in_video)
	{
		aifil::log_warning("GT first_frame: incorrect media type");
		return -1;
	}
	if (points.empty())
	{
		aifil::log_warning("GT first_frame: empty list");
		return -1;
	}

	return points.front().frame_num;
}

int PointList::last_frame() const
{
	if (!Results::in_video)
	{
		aifil::log_warning("GT last_frame: incorrect media type");
		return -1;
	}
	if (points.empty())
	{
		aifil::log_warning("GT last_frame: empty list");
		return -1;
	}

	return points.back().frame_num;
}


PointList::points_t::iterator PointList::create_point(const GroundTruthPoint &point)
{
	printf("PointList::create_point\n");
	points_t::iterator insert_position;

	if (!point.pic_name.empty())
		insert_position = points.insert(seek_first_larger(point.pic_name), point);
	else if (point.frame_num)
	{
		insert_position = points.insert(seek_first_larger(point.frame_num), point);
	}
	else
	{
		points.push_back(point);
		insert_position = points.end();
		--insert_position;
	}
	printf("new list was created\n");
	// new list was created
	if (points.size() == 1)
		current_pos = points.begin();
	return insert_position;
}

PointList::points_t::iterator PointList::update_point(const GroundTruthPoint &point)
{
	log_state("PointList::update_point");
	if (points.empty())
		return create_point(point);

	if (!point.pic_name.empty())
	{
		seek(point.pic_name);
		points_t::iterator it = current_pos;
		for ( ; it != points.end() && it->pic_name == point.pic_name; ++it)
		{
			if (it->object_id != point.object_id)
				continue;
			it->object_type = point.object_type;
			it->center_x = point.center_x;
			it->center_y = point.center_y;
			it->width = point.width;
			it->height = point.height;
			return it;
		}
	}
	else if (point.frame_num)
	{
		seek(point.frame_num);
		points_t::iterator it = current_pos;
		for ( ; it != points.end() && it->frame_num == point.frame_num; ++it)
		{
			if (it->object_id != point.object_id)
				continue;
			it->object_type = point.object_type;
			it->center_x = point.center_x;
			it->center_y = point.center_y;
			it->width = point.width;
			it->height = point.height;
			return it;
		}
	}
	else
		af_assert(!"incorrect frame for point updating");

	return create_point(point);
}

PointList::points_t::iterator PointList::erase_point(const GroundTruthPoint &point)
{
	log_state("PointList::erase_point\n");
	if (points.empty())
		return points.end();

	if (!point.pic_name.empty())
	{
		seek(point.pic_name);
		points_t::iterator it = current_pos;
		for ( ; it != points.end() && it->pic_name == point.pic_name; ++it)
		{
			if (it->object_id == point.object_id)
			{
				points.erase(it);
				current_pos = points.begin();
				return current_pos;
			}
		}
	}
	else if (point.frame_num)
	{
		for (auto it = points.begin(); it != points.end(); ++it)
		{
			aifil::log_state("lst %d frame %d\n",
							 it->object_id, it->frame_num);
		}
		seek(point.frame_num);
		points_t::iterator it = current_pos;
		for ( ; it != points.end() && it->frame_num == point.frame_num; ++it)
		{
			aifil::log_state("trying to find object %d in frame %d (%d %d)\n",
							 point.object_id, point.frame_num,
							 it->object_id, it->frame_num);
			if (it->object_id == point.object_id)
			{
				points.erase(it);
				current_pos = points.begin();
				return current_pos;
			}
		}
	}

	aifil::except(false,
		aifil::stdprintf("GT: can't erase point for object %d", point.object_id));
	return points.end();
}

PointList::points_t::iterator PointList::update_point(
	int frame_num, const std::string &frame_name,
	int object_id, int object_type,
	float new_x, float new_y, float new_w, float new_h
)
{
	GroundTruthPoint point;
	point.frame_num = frame_num;
	point.pic_name = frame_name;
	point.object_id = object_id;
	point.object_type = object_type;
	point.set_coords(new_x, new_y, new_w, new_h);

	return update_point(point);
}


bool PointList::read(const std::string &path)
{
	points.clear();

	setlocale(LC_NUMERIC, "C");
	FILE* file = 0;
	if (!path.empty())
		file = fopen(path.c_str(), "rb");
	if (!file)
		return false;

	int line = 0;
	while (!feof(file))
	{
		++line;
		GroundTruthPoint point;
		char buf[512];
		int r = fscanf(file, "%s %f %f %f %f %d %d ",
			buf,
			&(point.center_x), &(point.center_y), &(point.width), &(point.height),
			&(point.object_id), &(point.object_type));
		if (r < 7)
		{
			aifil::log_warning(
				"GT: file '%s' error on line %d: "
				"expected %d params, got %d",
				path.c_str(), line, 7, r);
			continue;
		}

		if (Results::in_video)
		{
			point.frame_num = atoi(buf);
			if (point.frame_num <= 0)
			{
				aifil::log_warning(
					"GT: file '%s' error on line %d: "
					"non-positive frame number (%d)",
					path.c_str(), line, point.frame_num);
				continue;
			}
		}
		else
			point.pic_name = buf;
		points.push_back(point);
	}
	fclose(file);

	points.sort();
	seek();
	return true;
}

bool PointList::write(const std::string &path, bool only_base)
{
	if (points.empty())
		return false;

	setlocale(LC_NUMERIC, "C");
	FILE* file = 0;
	if (!path.empty())
		file = fopen(path.c_str(), "wb");
	if (!file)
		return false;

	points.sort();
	int count = 0;
	for (points_t::iterator it = points.begin(); it != points.end(); ++it, ++count)
	{
		if (only_base && !it->is_base)
			continue;
		if (Results::in_video)
			fprintf(file, "%d\t", it->frame_num);
		else
			fprintf(file, "%s\t", it->pic_name.c_str());
		fprintf(file, "%5.2f\t%5.2f\t%5.2f\t%5.2f\t%d\t%d\n",
			it->center_x, it->center_y, it->width, it->height, it->object_id, it->object_type);
	}
	fclose(file);
	aifil::log_state("GT: %d points written to '%s'", count, path.c_str());
	return true;
}

bool Results::read_points(const std::string &path, bool to_gt, bool want_tracks)
{
	if (to_gt)
		return read_points(path, all_points_correct/*, sample_tracks*/, want_tracks);
	else
		return read_points(path, all_points_detected/*, found_tracks*/, want_tracks);
}

bool Results::read_points(const std::string &path, PointList &all_points,
	/*tracks_t &tracks,*/ bool want_tracks)
{
	aifil::log_state("GT: loading from '%s'", path.c_str());
	//tracks.clear();
	if (!all_points.read(path))
		return false;

	aifil::log_state("GT: loaded %d points", all_points.points.size());
	/*
	if (!want_tracks)
		return true;

	//cluster points by IDs
	for (PointList::points_t::iterator it = all_points.points.begin();
			it != all_points.points.end(); ++it)
		tracks[it->object_id].points.push_back(*it);

	for (tracks_t::iterator it = tracks.begin(); it != tracks.end(); ++it)
		it->second.seek();

	aifil::log_state("GT: loaded %d tracks", tracks.size());
	*/
	return true;
}

void Results::clear_points(bool gt)
{
	if (gt)
	{
		all_points_correct.points.clear();
		//sample_tracks.clear();
	}
	else
	{
		last_detection = 0;
		all_points_detected.points.clear();
		//found_tracks.clear();
	}
}

void Results::skip_frame(int frame_num)
{
	all_points_correct.seek(frame_num);
	if (!all_points_correct.points.empty() &&
		all_points_correct.current_pos != all_points_correct.points.end() &&
		all_points_correct.current_pos->frame_num == frame_num)
		all_points_correct.current_pos->skipped = true;

	/*for (std::map<int, PointList>::iterator it = sample_tracks.begin();
		it != sample_tracks.end(); ++it)
	{
		PointList &sample_track = it->second;
		if (sample_track.points.empty())
			continue;
		if (sample_track.current_pos == sample_track.points.end())
			continue;

		GroundTruthPoint &sample_point = *(sample_track.current_pos);
		sample_track.seek(frame_num);
		if (frame_num == sample_point.frame_num)
			sample_point.skipped = true;
	}
	*/
}

void Results::add_detections(const std::vector<ResultDetection> &objects,
		int frame_num, const std::string &frame_name)
{
	for (size_t i = 0; i < objects.size(); ++i)
	{
		const ResultDetection &f = objects[i];

		GroundTruthPoint point;
		point.frame_num = frame_num;
		point.pic_name = frame_name;
		point.center_x = f.center_x;
		point.center_y = f.center_y;
		point.width = f.width;
		point.height = f.height;
		point.object_id = ++last_detection;
		point.object_type = f.type;

		if (!frame_name.empty())
			all_points_detected.points.insert(
				all_points_detected.seek_first_larger(frame_name), point
			);
		else if (frame_num)
			all_points_detected.points.insert(
				all_points_detected.seek_first_larger(frame_num), point
			);
		else
			all_points_detected.points.push_back(point);
	}
}

void Results::add_tracks(const std::vector<ResultTarget> &objects, int frame_num)
{
	for (size_t i = 0; i < objects.size(); ++i)
	{
		const ResultTarget &f = objects[i];

		GroundTruthPoint point;
		point.frame_num = frame_num;
		point.center_x = f.center_x;
		point.center_y = f.center_y;
		point.width = f.width;
		point.height = f.height;
		point.object_id = f.id;
		point.object_type = f.type;

		all_points_detected.points.insert(
			all_points_detected.seek_first_larger(frame_num), point
		);
		//found_tracks[f.id].points.push_back(point);
	}
}

void Results::collect_det_point_stat(int obj_type, DetectorStat &stat,
	std::vector<GroundTruthPoint *> &gt, std::vector<GroundTruthPoint *> &res)
{
	for (int gi = 0; gi < (int)gt.size(); ++gi)
		gt[gi]->clear_stat();
	for (int ri = 0; ri < (int)res.size(); ++ri)
		res[ri]->clear_stat();

	for (int ri = 0; ri < (int)res.size(); ++ri)
	{
		if (obj_type && res[ri]->object_type != obj_type)
			continue;

		for (int gi = 0; gi < (int)gt.size(); ++gi)
		{
			if (obj_type && gt[gi]->object_type != obj_type)
				continue;

			float sim = gt_point_similarity(*res[ri], *gt[gi]);
			if (sim > det_followers_th)
			{
				++res[ri]->followers;
				++gt[gi]->followers;
			}
			if (sim > gt[gi]->found_percent)
				gt[gi]->found_percent = sim;
			if (sim > res[ri]->found_percent)
				res[ri]->found_percent = sim;
		}
	}

	for (int i = 0; i < (int)gt.size(); ++i)
	{
		const GroundTruthPoint *pt = gt[i];
		if (obj_type && pt->object_type != obj_type)
			continue;

		++stat.gt_objects;
		stat.gt_followers += pt->followers;
		if (pt->followers)
			++stat.gt_follow_pts;
		if (pt->found_percent > det_metric_same)
			++stat.fully_found;
		else if (pt->found_percent > det_metric_good)
			++stat.partially_found;
		else if (pt->found_percent > det_metric_bad)
			++stat.something_found;
		else
			++stat.not_found;
	}

	for (int i = 0; i < (int)res.size(); ++i)
	{
		const GroundTruthPoint *pt = res[i];
		if (obj_type && pt->object_type != obj_type)
			continue;

		++stat.detections;
		stat.res_followers += pt->followers;
		if (pt->followers)
			++stat.res_follow_pts;
		if (pt->found_percent > det_metric_same)
			++stat.fully_correct;
		else if (pt->found_percent > det_metric_good)
			++stat.partially_correct;
		else if (pt->found_percent > det_metric_bad)
			++stat.something_correct;
		else
			++stat.junk;
	}
}

bool Results::collect_detector_stat(int obj_type, DetectorStat &stat)
{
	if (all_points_correct.points.empty() || all_points_detected.points.empty())
	{
		aifil::log_state("STAT: no detections");
		return false;
	}

	if (in_video)
		collect_detector_stat_movie(obj_type, stat);
	else
		collect_detector_stat_images(obj_type, stat);
	return true;
}

void Results::collect_detector_stat_movie(int obj_type, DetectorStat &stat)
{
	int max_frame = all_points_detected.points.back().frame_num;
	if (all_points_correct.points.back().frame_num > max_frame)
		max_frame = all_points_correct.points.back().frame_num;

	stat.frames = max_frame;
	all_points_correct.seek();
	all_points_detected.seek();
	for (int fr = 0; fr < max_frame; ++fr)
	{
		std::vector<GroundTruthPoint*> gt = all_points_correct.select_by_frame(fr);
		std::vector<GroundTruthPoint*> res = all_points_detected.select_by_frame(fr);
		collect_det_point_stat(obj_type, stat, gt, res);
	}
}

void Results::collect_detector_stat_images(int obj_type, DetectorStat &stat)
{
	std::set<std::string> all_frames;
	for (auto it = all_points_detected.points.begin();
		 it != all_points_detected.points.end(); ++it)
		all_frames.insert(it->pic_name);
	for (auto it = all_points_correct.points.begin();
		 it != all_points_correct.points.end(); ++it)
		all_frames.insert(it->pic_name);

	stat.frames = int(all_frames.size());
	all_points_correct.seek();
	all_points_detected.seek();
	for (auto it = all_frames.begin(); it != all_frames.end(); ++it)
	{
		std::vector<GroundTruthPoint*> gt = all_points_correct.select_by_frame(*it);
		std::vector<GroundTruthPoint*> res = all_points_detected.select_by_frame(*it);
		collect_det_point_stat(obj_type, stat, gt, res);
	}
}

void Results::save_detector_stat(const std::string &path, int obj_type)
{
	DetectorStat stat;
	if (!collect_detector_stat(obj_type, stat))
		return;

	int hg_rev = hg_revision();
	std::string filename = path;
	if (hg_rev)
		filename += aifil::stdprintf(".%d", hg_rev);
	FILE *file = fopen(filename.c_str(), "wb");
	if (!file)
	{
		aifil::log_warning("Cannot save detector stat!");
		return;
	}

	fprintf(file, "hg revision: %d\n", hg_rev);
	if (!comment.empty())
		fprintf(file, "%s\n", comment.c_str());
	else
		fprintf(file, "\n");

	fprintf(file, "GT objects (total):           %d\n", stat.gt_objects);
	fprintf(file, "Fully found (>%.2f):          %.2lf%% (%d)\n",
		det_metric_same, stat.fully_found * 100.0 / stat.gt_objects,
		stat.fully_found);
	fprintf(file, "Partially found (>%.2f):      %.2lf%% (%d)\n",
		det_metric_good, stat.partially_found * 100.0 / stat.gt_objects,
		stat.partially_found);
	fprintf(file, "Something found (>%.2f):      %.2lf%% (%d)\n",
		det_metric_bad, stat.something_found * 100.0 / stat.gt_objects,
		stat.something_found);
	fprintf(file, "Not found (<=%.2f):           %.2lf%% (%d)\n",
		det_metric_bad, stat.not_found * 100.0 / stat.gt_objects,
		stat.not_found);
	fprintf(file, "Detections per point:         %.2lf\n\n",
		stat.gt_follow_pts ? double(stat.gt_followers) / stat.gt_follow_pts : 0.0);

	fprintf(file, "Detections (total):           %d\n", stat.detections);
	fprintf(file, "Fully correct (>%.2f):        %.2lf%% (%d)\n",
		det_metric_same, stat.fully_correct * 100.0 / stat.detections,
		stat.fully_correct);
	fprintf(file, "Partially correct (>%.2f):    %.2lf%% (%d)\n",
		det_metric_good, stat.partially_correct * 100.0 / stat.detections,
		stat.partially_correct);
	fprintf(file, "Something correct (>%.2f):    %.2lf%% (%d)\n",
		det_metric_bad, stat.something_correct * 100.0 / stat.detections,
		stat.something_correct);
	fprintf(file, "Incorrect (<=%.2f):           %.2lf%% (%d)\n",
		det_metric_bad, stat.junk * 100.0f / stat.detections, stat.junk);
	fprintf(file, "GTs per point:                %.2lf\n\n",
		stat.res_follow_pts ? double(stat.res_followers) / stat.res_follow_pts : 0.0);

	int tp = stat.fully_found;
	int fp = stat.junk;
	fprintf(file, "Recall (tp / (tp + missed)):  %.2lf%%\n",
		tp + stat.not_found ? tp * 100.0 / (tp + stat.not_found) : 0.0);
	fprintf(file, "Precision (tp / (tp + fp)):   %.2lf%%\n",
		tp + fp ? tp * 100.0 / (tp + fp) : 0.0);
	fprintf(file, "FPPI:                         %.2lf\n\n",
		double(fp) / stat.frames);

	fclose(file);
}

bool Results::collect_tracker_stat(int max_frame)
{
/*
	if (sample_tracks.empty() || found_tracks.empty())
	{
		aifil::log_state("STAT: no tracks");
		return false;
	}

	for (std::map<int, PointList>::iterator it = sample_tracks.begin();
		it != sample_tracks.end(); ++it)
	{
		af_assert(!it->second.points.empty());
		it->second.seek();
	}
	for (std::map<int, PointList>::iterator it = found_tracks.begin();
		it != found_tracks.end(); ++it)
	{
		af_assert(!it->second.points.empty());
		it->second.seek();
	}


	// float size_th = 0.5f; //cross-correlation

	if (max_frame == -1)
	{
		max_frame = all_points_detected.points.back().frame_num;
		if (all_points_correct.points.back().frame_num > max_frame)
			max_frame = all_points_correct.points.back().frame_num;
	}

	for (int fr = 0; fr < max_frame; ++fr)
	{
		for (std::map<int, PointList>::iterator fit = found_tracks.begin();
			fit != found_tracks.end(); ++fit)
		{
			PointList &current_track = fit->second;
			current_track.seek(fr);

			//track is fully processed
			if (current_track.current_pos == current_track.points.end())
				continue;

			GroundTruthPoint &pt = *(current_track.current_pos);
			if (fr != pt.frame_num) //track is in the future
				continue;

			for (tracks_t::iterator sit = sample_tracks.begin(); sit != sample_tracks.end(); ++sit)
			{
				PointList &sample_track = sit->second;
				sample_track.seek(fr);

				//track is fully processed
				if (sample_track.current_pos == sample_track.points.end())
					continue;

				GroundTruthPoint &spt = *(sample_track.current_pos);
				if (fr != spt.frame_num) //track is in the future
					continue;

				float dist = (pt.center_x - spt.center_x) * (pt.center_x - spt.center_x) +
						(pt.center_y - spt.center_y) * (pt.center_y - spt.center_y);
				// float size_diff = 1.0f;
				float dist_th = std::max(spt.width, spt.height) / 2;

				if (dist > dist_th * dist_th)
					continue;

				pt.found = true;
				spt.found = true;

				if (sample_track.points_per_id.find(fit->first) == sample_track.points_per_id.end())
					sample_track.points_per_id[fit->first] = 1;
				else
					sample_track.points_per_id[fit->first]++;

				if (current_track.points_per_id.find(sit->first) == current_track.points_per_id.end())
					current_track.points_per_id[sit->first] = 1;
				else
					current_track.points_per_id[sit->first]++;
			} //for each sample track
		} //for each found track
	} //for each frame

	for (std::map<int, PointList>::iterator it = sample_tracks.begin();
		it != sample_tracks.end(); ++it)
	{
		PointList &track = it->second;
		track.found_percent = float(track.points_found()) / track.points_not_skipped();
	}
	for (std::map<int, PointList>::iterator it = found_tracks.begin();
		it != found_tracks.end(); ++it)
	{
		PointList &track = it->second;
		track.found_percent = float(track.points_found()) / track.points_not_skipped();
	}
*/
	return true;
}

void Results::save_tracker_stat(const std::string &path)
{
	if (!collect_tracker_stat())
		return;
/*
	float metric_same = 0.8f;
	float metric_good = 0.5f;
	float metric_bad = 0.2f;

	int fully_found = 0;
	int fully_covered = 0;
	int partially_found = 0;
	int something_found = 0;
	int not_found = 0;

	int fully_correct = 0;
	int composite = 0;
	int partially_correct = 0;
	int something_correct = 0;
	int junk = 0;
*/
/*
	std::string corr_track_info;
	for (std::map<int, PointList>::iterator trk = sample_tracks.begin();
		trk != sample_tracks.end(); ++trk)
	{
		PointList &sample_track = trk->second;
		corr_track_info += aifil::stdprintf("%3d (%3d p): ",
			trk->first, int(sample_track.points.size()));
		int max_points = 0;
		std::string track_info;
		int followers = 0;
		int points_total = sample_track.points_not_skipped();
		for (std::map<int, int>::iterator it = sample_track.points_per_id.begin();
				it != sample_track.points_per_id.end(); ++it)
		{
			int points_num = it->second;
			if (points_num > max_points)
				max_points = points_num;
			float trk_percent = points_num * 100.0f / points_total;
			if (trk_percent < 5.0f)
				continue;

			if (followers != 0)
				track_info += aifil::stdprintf(", ");
			track_info += aifil::stdprintf("id %d - %.2f%%", it->first, trk_percent);
			++followers;
		}
		if (!track_info.empty())
			track_info = "(" + track_info + ")";
		float percent = sample_track.found_percent;
		float longest_percent = (float)(max_points) / points_total;
		corr_track_info += aifil::stdprintf("%.2f%% (longest %.2f%%) covered by %d tracks %s\n",
			percent * 100.0f, longest_percent * 100.0f, followers, track_info.c_str());

		if (longest_percent > metric_same)
			++fully_found;
		else if (percent > metric_same)
			++fully_covered;
		else if (percent > metric_good)
			++partially_found;
		else if (percent > metric_bad)
			++something_found;
		else
			++not_found;
	}

	std::string found_track_info;
	for (std::map<int, PointList>::iterator trk = found_tracks.begin();
		trk != found_tracks.end(); ++trk)
	{
		PointList &found_track = trk->second;
		found_track_info += aifil::stdprintf("%3d (%3d p): ",
			trk->first, int(found_track.points.size()));
		int max_points = 0;
		std::string track_info;
		int followers = 0;
		int points_total = found_track.points_not_skipped();
		for (std::map<int, int>::iterator it = found_track.points_per_id.begin();
				it != found_track.points_per_id.end(); ++it)
		{
			int points_num = it->second;
			if (points_num > max_points)
				max_points = points_num;
			float trk_percent = points_num * 100.0f / points_total;
			if (trk_percent < 5.0f)
				continue;

			if (followers != 0)
				track_info += aifil::stdprintf(", ");
			track_info += aifil::stdprintf("id %d - %.2f%%", it->first, trk_percent);
			++followers;
		}
		if (!track_info.empty())
			track_info = "(" + track_info + ")";
		float percent = found_track.found_percent;
		float longest_percent = (float)(max_points) / points_total;
		found_track_info += aifil::stdprintf("%.2f%% (longest %.2f%%) covered by %d tracks %s\n",
			percent * 100.0f, longest_percent * 100.0f, followers, track_info.c_str());

		if (longest_percent > metric_same)
			++fully_correct;
		else if (percent > metric_same)
			++composite;
		else if (percent > metric_good)
			++partially_correct;
		else if (percent > metric_bad)
			++something_correct;
		else
			++junk;
	}
*/
	int hg_rev = hg_revision();
	std::string filename = path;
	if (hg_rev)
		filename += aifil::stdprintf(".%d", hg_rev);
	FILE *file = fopen(filename.c_str(), "wb");
	if (!file)
	{
		aifil::log_warning("Cannot save tracker stat!");
		return;
	}

	fprintf(file, "hg revision: %d\n", hg_rev);
	if (!comment.empty())
		fprintf(file, "%s\n", comment.c_str());
	else
		fprintf(file, "\n");

/*
	fprintf(file, "Real tracks (total):          %d\n",
		int(sample_tracks.size()));
	fprintf(file, "Fully found (>%.2f):          %.2f%% (%d)\n",
		metric_same, fully_found * 100.0f / sample_tracks.size(), fully_found);
	fprintf(file, "Fully covered (>1 followers): %.2f%% (%d)\n",
		fully_covered * 100.0f / sample_tracks.size(), fully_covered);
	fprintf(file, "Partially found (>%.2f):      %.2f%% (%d)\n",
		metric_good, partially_found * 100.0f / sample_tracks.size(), partially_found);
	fprintf(file, "Something found (>%.2f):      %.2f%% (%d)\n",
		metric_bad, something_found * 100.0f / sample_tracks.size(), something_found);
	fprintf(file, "Not found (<=%.2f):           %.2f%% (%d)\n\n",
		metric_bad, not_found * 100.0f / sample_tracks.size(), not_found);

	fprintf(file, "Found tracks (total):         %d\n",
		int(found_tracks.size()));
	fprintf(file, "Fully correct (>%.2f):        %.2f%% (%d)\n",
		metric_same, fully_correct * 100.0f / found_tracks.size(), fully_correct);
	fprintf(file, "Good composite (>1 tracks):   %.2f%% (%d)\n",
		composite * 100.0f / found_tracks.size(), composite);
	fprintf(file, "Partially correct (>%.2f):    %.2f%% (%d)\n",
		metric_good, partially_correct * 100.0f / found_tracks.size(), partially_correct);
	fprintf(file, "Something correct (>%.2f):    %.2f%% (%d)\n",
		metric_bad, something_correct * 100.0f / found_tracks.size(), something_correct);
	fprintf(file, "Incorrect (<=%.2f):           %.2f%% (%d)\n\n",
		metric_bad, junk * 100.0f / found_tracks.size(), junk);

	fprintf(file, "---------------------------------------------\n");
	fprintf(file, "Correct tracks recognition:\n%s", corr_track_info.c_str());
	fprintf(file, "\nFound tracks quality:\n%s", found_track_info.c_str());
*/
	fclose(file);
}

void Results::save_time_stat(const std::string &path, double execution_time, int frames)
{
	int hg_rev = hg_revision();
	std::string filename = path;
	if (hg_rev)
		filename += aifil::stdprintf(".%d", hg_rev);
	FILE *file = fopen(filename.c_str(), "wb");
	if (!file)
	{
		aifil::log_warning("Cannot save performance stat!");
		return;
	}
	fprintf(file, "hg revision: %d\n", hg_rev);
	if (!comment.empty())
		fprintf(file, "%s\n", comment.c_str());
	else
		fprintf(file, "\n");

	fprintf(file, "Execution Time:      %s\n", aifil::sec2time(int(execution_time)).c_str());
	fprintf(file, "Per frame:           %lf ms\n\n", execution_time * 1e3 / frames);
	fprintf(file, "%s", aifil::Profiler::instance().print_statistics().c_str());
	fclose(file);
}

void Results::write_stat(const Task &task, double execution_time, int frames)
{
	save_detector_stat(task.result_splitext_0() + ".det_stat", task.object_type);
	save_tracker_stat(task.result_splitext_0() + ".track_stat");
	save_time_stat(task.result_splitext_0() + ".time_stat", execution_time, frames);
	log_state("statistics was written in '%s.*'", task.result_splitext_0().c_str());
}

}  // namespace ground_truth
