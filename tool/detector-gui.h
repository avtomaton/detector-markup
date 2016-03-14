#ifndef GT_DETECTOR_GUI_H
#define GT_DETECTOR_GUI_H

#include <video/media-reader.h>
#include "gt-utils.h"

#include <opencv2/opencv.hpp>

#include <QtCore/qglobal.h>
#if (QT_VERSION >= QT_VERSION_CHECK(5, 0, 0))
#include <QtCore/QTimer>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QLabel>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QMessageBox>
#include <QtWidgets/QTableWidget>
#include <QtGui/QPainter>
#include <QtGui/QWheelEvent>
#else
#include <QtGui/QtGui>
#endif

namespace Ui {
class DemoWindow;
}

namespace aifil {
struct SequentalReader;
struct MatCache;
}
using aifil::SequentalReader;
using aifil::MatCache;

namespace ground_truth {

//class DetHandler;

struct GUITrackPoint : public GroundTruthPoint
{
	GUITrackPoint() : table_row(-1) {}
	GUITrackPoint(float cx, float cy, float w, float h)
		: GroundTruthPoint(cx, cy, w, h),
		  table_row(-1)
	{}

	int table_row;
};

struct GUITrack : public ObjectTrack
{
	GUITrack(const GroundTruthPoint &pt, ObjectTrack::INTERPOLATION interp)
		: ObjectTrack(pt, interp),
		  table_row(-1) {}

	int table_row;
};

class DetWindow: public QWidget
{
	Q_OBJECT

public:
	DetWindow(QWidget* parent);
	~DetWindow();

	enum PLAY_MODE { PLAY, PAUSE, PLAY_PAUSE, PLAY_FAST };
	void reset_all();
	void reset(const std::string &path, double wanted_fps = 25.0);
	void set_play_mode(PLAY_MODE new_mode);
	void ensure_visible(int frame_first, int frame_last);
	void read_cur_frame();
	bool get_frame(int frames_from_cur); //return false if last frame
	bool seek(int dir, bool emit_signal = true);//gt-editor

	void next_frame();//gt-editor
	void prev_frame();//gt-editor
	void start_movie();//gt-editor
	void end_movie();//gt-editor
	void jump_frame();//gt-editor

	int cur_frame_num;
	QString cur_frame_name();
	int cur_frame_w();
	int cur_frame_h();

	void compute_frame();
	void detections_show(bool state);
	void targets_show(bool state);
	void results_show(bool state);
	void gt_show(bool state);
	void zones_show(bool state);
	void lost_objects_show(bool);
	void feature_tracks_show(bool);
	void lost_features_show(bool);
	void prefilter_use(bool);
	void bg_mask_show(bool);
	void read_type_name();

	static const int TRACE_FRAMES_NUM;
	static const int JUMP_FRAMES;
private:
	static const int CACHE_MAX_FRAMES;

	PLAY_MODE movie_play_mode;
	SequentalReader *media_seq;
	typedef std::map<int, QImage> frame_archive_t;
	frame_archive_t frame_archive;

	// visibility settings
	bool is_detections_visible;
	bool is_gt_visible;
	bool is_targets_visible;
	bool is_results_visible;
	bool is_zones_visible;

	//DetHandler *det_handler;

	QTimer *timer;
	double fps;
	std::string address;

	bool is_rect_move;
	enum { NONE = 0, LEFT_TOP, TOP,
		RIGHT_TOP, LEFT, RIGHT,
		LEFT_BOTTOM, BOTTOM, RIGHT_BOTTOM } mouse_stretch_mode;

	std::vector<DetectorZoneParams> zones;

private:
	void zones_update();

public:

	int edited_track;
	int edited_type;

	enum WORKMODE
	{
		MODE_NOTHING = 0,
		MODE_DETECT = 0x1,
		MODE_TRACK = 0x2,
		MODE_ZONE_CREATE = 0x4,
		MODE_CORRECT_CREATE = 0x8,
		MODE_CROP = 0x10,
		MODE_CLASSIFY = 0x20,
		MODE_GROUND_PLANE = 0x40
	};

	int workmode;

	void paintEvent(QPaintEvent *ev);
	void mousePressEvent(QMouseEvent *ev);
	void mouseReleaseEvent(QMouseEvent *ev);
	void mouseMoveEvent(QMouseEvent *ev);
	void kev(QKeyEvent *);

	std::string zone_selected(int zone);
	void zone_type_changed(int zone, const std::string &type);

	void obj_params_to_ui();
	void ui_to_obj_params(double w, double h);

signals:
	void mouse_label_changed(const QString &text);
	void frame_num_changed(const QString &text);
	void obj_size_changed(int w, int h);
	void ready();
	void gt_track_added(int track_id);
	void gt_track_changed(int track_id);
	void gt_track_removed(int track_id, int row_in_table);
	void gt_track_selected();
	void detected_track_changed();
public slots:
	void timeout();
	void set_workmode(int mode);

public:
	//simt2::Detector *simt_det;
	//human::PeopleCounter *people_counter;

	cv::Rect crop_rect;
	QPixmap frame_pixmap;

	bool mlb_down;
	bool draw_traces;

	int mouse_x, mouse_y;
	QPoint old_pos;
	QPoint old_pos_dist;

	std::vector<ResultTarget> objects;
	QRgb feature_color;

	int active_zone;
	bool new_frame_arrived;
	int last_computed_frame;
	//int visualize_bg;

	typedef std::map<int, GUITrack> tracks_t;
	tracks_t gt_tracks;
	tracks_t found_tracks;

	GUITrack *cur_sample_track;
	GUITrack *cur_found_track;
	GroundTruthPoint *cur_sample_point;

	GroundTruthPoint last_gt_point;

	QMap<int, QString> frame_names;

	/*
	void update_size_map_point(bool horizontal, bool minimal, double delta);
	*/
	GUITrack make_track(const GroundTruthPoint &pt);
	void update_result_point(int obj_id, int obj_type,
		float new_x, float new_y, float new_w = -1, float new_h = -1);
	void results2tracks();
	void convert_name2int();
	void del_current_track();
	void del_current_point();
	GroundTruthPoint* get_current_point();

	void update_current_point();
	void move_rect(int dx, int dy, bool update_lists = true);
	void stretch_rect(int dx, int dy, bool update_lists = true);
	void change_cur_point_size(float w, float h);
	void stretch_with_mode(int dx, int dy, GroundTruthPoint *ob);

	void add_sample_track(int obj_type,
		const GroundTruthPoint &pt = GroundTruthPoint());
	QRect get_rect_on_widget(const GUITrack &track, int frame);

	void picture_settings_update();
	void visualize();

	void draw_logic_rect(const std::vector<GroundTruthPoint*> &vec,
		float margin, QPainter *painter,
		Qt::GlobalColor color0 = Qt::transparent,
		Qt::GlobalColor color1 = Qt::transparent);
	void draw_track(const GUITrack &track, QPainter *painter, bool active);
	void draw_track_trace(const GUITrack &track,
		QPainter* painter, QColor color);
};

class DemoWindow: public QMainWindow
{
	Q_OBJECT
public:
	DemoWindow();
	~DemoWindow();

private:
	Ui::DemoWindow *ui;
	QLabel *mouse_label;
	QLabel *track_label;
	QLabel *frame_label;

	DetWindow *det_window;
	typedef std::map<int, std::string> subtype_name_t;
	typedef std::map<int, std::string> type_name_t;
	typedef std::map<int, subtype_name_t> type_map_t;
	type_name_t type_name;
	type_map_t type_map;
	int type_split_bit;
public slots:
	void keyPressEvent(QKeyEvent* kev);
	void wheelEvent(QWheelEvent *);
	void table_update(int track_id);
	void table_remove_row(int, int row);
	void table_reload(int track_id);
	void update_cur_type();
private slots:
	void renew_detector_stat();

	void set_mouse_label(const QString &text);
	void set_frame_label(const QString &text);
	void set_object_size(int w, int h);
	void on_mode_currentIndexChanged(int index);
	void on_play_button_clicked();
	void on_prev_button_clicked();
	void on_next_button_clicked();
	void on_jump_button_clicked();
	void on_fast_button_clicked();

	void on_zone_number_currentIndexChanged(int index);
	void on_zone_type_currentIndexChanged(int index);

	void on_gt_obj_table_clicked(const QModelIndex &index);
	void on_result_obj_table_clicked(const QModelIndex &index);

	void on_table_track_view_clicked();
	void on_table_point_view_clicked();

	void on_object_type_currentIndexChanged(const QString &);
	void on_object_subtype_currentIndexChanged(const QString &);

	//void on_object_type_editingFinished();
	void on_obj_size_w_valueChanged(double new_val);
	void on_obj_size_h_valueChanged(double new_val);

	void on_open_task_act_triggered();
	void on_save_gt_act_triggered();
	void on_load_gt_act_triggered();
	void on_load_result_act_triggered();
	void on_save_result_act_triggered();

	void on_gt_checkbox_toggled(bool checked);
	void on_results_checkbox_toggled(bool checked);
	void on_detections_checkbox_toggled(bool checked);
	void on_targets_checkbox_toggled(bool checked);
	void on_zones_checkbox_toggled(bool checked);

	void on_exit_act_triggered();
	void on_help_act_triggered();
	void on_about_act_triggered();
	void on_compute_stat_act_triggered();

public:
	void read_new_task(const std::string &task_file_name);
private:
	void read_type_name();
	void update_subtypes(int subtype);
	int type_number(const std::string &type);
	int subtype_number(int inttype, const std::string &subtype);

	void type_string(int typenumber, std::string &type, std::string &subtype);

	void type_split(int typenumber, int *type, int *subtype);
	int type_merge(int type, int subtype);

	void connector();
	void reset_gt();
	bool table_track_update(const GUITrack &track, QTableWidget *ui_table);
	bool table_track_remove(int row, QTableWidget *ui_table);
	bool table_point_update(const GUITrackPoint &point, QTableWidget *ui_table);
	void objects_to_ui(DetWindow::tracks_t &tracks, QTableWidget *ui_table);
	void objects_to_ui(PointList &data, QTableWidget *ui_table);
	void tables_reload();
};

}  // namespace ground_truth

#endif // GT_DETECTOR_GUI_H
