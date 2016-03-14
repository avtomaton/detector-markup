#ifndef AIFIL_GT_UTILS_H
#define AIFIL_GT_UTILS_H

#include <common/conf-parser.h>

#include "dump-utils.h"
#include "video/media-reader.h"

#include <opencv2/core/core.hpp>

#include <stdint.h>
#include <list>
#include <vector>

namespace aifil {
struct MatCache;
}

namespace anfisa {
struct ResultTarget;
}
using anfisa::ResultTarget;

namespace ground_truth {

const std::string gt_base_ext = ".gt";
const std::string gt_trusted_ext = ".correct";
const std::string gt_all_ext = ".points";
const std::string gt_det_points_ext = ".result";
const std::string gt_det_dump_ext = ".det_dump";
const std::string gt_icf_dump_ext = ".dump";
const std::string gt_icf_desc_ext = ".desc";

// forward declaration for ObjectTrack
struct PointList;

struct Task : aifil::ConfigParser
{
	std::string action;  // test, dump

	// source
	std::string movie_path;
	std::string photo_dir;

	// paths
	// if path is "file or folder", it means that both file and folder
	// can be specified in configuration file, but after reading
	// path to file will be calculated:
	// if folder was specified, file would be constructed using task name
	std::string settings_path;  // file
	std::string correct_path;  // file or folder
	std::string correct_splitext_0() const;
	std::string result_path;  // file or folder
	std::string result_splitext_0() const;
	std::string export_path;  // folder
	std::string dump_path;  // folder

	std::string obj_type_legend;

	// frame parameters
	int resize_w;
	int resize_h;
	int wanted_fps;
	int frames_cache_size;
	int stop_frame;
	int skip_frames;
	int detector_skip_frames;
	std::string colorspace;

	std::string worker_type;

	std::string ground_plane;
	std::string border_coords;

	//gt params
	int object_type;

	//dump params
	std::string dump_type;  // icf_good, icf_dssl, icf_canonical, head_det
	std::string dump_prefix;
	int dump_images;
	int dump_pedantic_size;
	std::list<DumpStatICF> dump_tile_params;

	std::string interpolation;  // linear, polynom
	int interpolation_points;

	Task();
	void defaults();
	void cook();
	bool is_movie();

	void prepare_to_work(const std::string &task_name);

	//callbacks
	static void resize_parse(ConfigParser* myself, const std::string &val);
	static void erase_file(ConfigParser* myself, const std::string &val);
	static void ground_plane_parse(ConfigParser* myself, const std::string &val);
	static void dump_tile_parse(ConfigParser* myself, const std::string &val);

private:
	Task(const Task &) : aifil::ConfigParser() {} //restrict copying
};

struct GroundTruthPoint
{
	std::string pic_name;
	int frame_num;

	float center_x;
	float center_y;
	float width;
	float height;

	int object_id;
	int object_type;

	bool is_base;
	bool skipped;

	// statistics
	bool found;
	int followers;
	float found_percent;

	GroundTruthPoint();
	GroundTruthPoint(float cx, float cy, float w, float h);
	bool empty() const;
	bool operator<(const GroundTruthPoint &other);
	void set_coords(float new_x = -1, float new_y = -1, float new_w = -1, float new_h = -1);
	cv::Rect rect(int frame_w = 100, int frame_h = 100) const;
	void set_rect(const cv::Rect &rect, int frame_w, int frame_h);

	void clear_stat();

	static GroundTruthPoint interp_linear(
		const GroundTruthPoint &ob0, const GroundTruthPoint &ob1, int frame_cur);
	static GroundTruthPoint interp_polynom(
		const std::vector<GroundTruthPoint> &obs, int frame_cur, int count_points = 2);
};

struct ObjectTrack
{
	enum POINT_TYPE {
		POINT_ABSENT,
		POINT_BASE,
		POINT_INTERPOLATED,
		POINT_EXTRAPOLATED
	};

	enum INTERPOLATION {
		INTERP_LINEAR,
		INTERP_POLYNOM
	};

	ObjectTrack(const GroundTruthPoint &point, INTERPOLATION interp = INTERP_LINEAR);

	INTERPOLATION interpolation;
	int interpolation_points;

	typedef std::map<int, GroundTruthPoint> points_t;
	points_t points;
	int object_id;
	int object_type;
	bool extrapolation_enable;
	int start_frame;
	int finish_frame;

	// statistics
	float found_percent;
	std::map<int, int> points_per_id; //ideal is 1 detected object with points.size() points

	GroundTruthPoint extrapolate(int frame) const;
	GroundTruthPoint get_point(int frame) const;
	POINT_TYPE point_type(int frame) const;

	int points_base() const;
	int points_found() const;
	int points_not_skipped() const;

	void update();
	void update_type(int new_type, PointList *pt_list = 0);
	void update_base_pt(GroundTruthPoint ob);
	void find_near_frames(int frame, int& frame_low, int& frame_high) const;
};

class PointList
{
public:
	PointList();

	typedef std::list<GroundTruthPoint> points_t;
	points_t points;
	points_t::iterator current_pos;

	//statistics
	float found_percent;
	std::map<int, int> points_per_id; //ideal is 1 detected object with points.size() points

	//by default set position to beginning of list
	void seek(); //go to the beginning
	void seek(int frame_num);
	void seek(const std::string &frame_name);

	points_t::iterator seek_first_larger(int frame_num);
	points_t::iterator seek_first_larger(const std::string &frame_name);

	// selects
	std::vector<GroundTruthPoint*> select_by_frame(int frame_num);
	std::vector<GroundTruthPoint*> select_by_frame(const std::string &frame_name);

	int points_found() const;
	int points_not_skipped() const;
	int first_frame() const;
	int last_frame() const;

	// updates
	points_t::iterator update_point(const GroundTruthPoint &point);
	points_t::iterator erase_point(const GroundTruthPoint &point);

	points_t::iterator update_point(int frame_num, const std::string &frame_name,
			int object_id, int object_type = 1,
			float new_x = -1, float new_y = -1, float new_w = -1, float new_h = -1);

	bool read(const std::string &path);
	bool write(const std::string &path, bool only_base = false);

private:
	points_t::iterator create_point(const GroundTruthPoint &point);
};

struct DetectorStat
{
	DetectorStat()
		: frames(0),
		  detections(0), gt_objects(0),
		  gt_followers(0), gt_follow_pts(0), res_followers(0), res_follow_pts(0),
		  fully_found(0), partially_found(0), something_found(0), not_found(0),
		  fully_correct(0), partially_correct(0), something_correct(0), junk(0)
	{}

	int frames;

	int detections;
	int gt_objects;

	int gt_followers; //sum of followers for each follow_pt
	int gt_follow_pts;
	int res_followers;
	int res_follow_pts;

	int fully_found;
	int partially_found;
	int something_found;
	int not_found;

	int fully_correct;
	int partially_correct;
	int something_correct;
	int junk;
};

struct Results
{
	Results();

	std::string comment;
	static bool in_video;
	int last_detection;

	//all points sorted by frame number
	PointList all_points_correct;
	PointList all_points_detected;

	//tracks of points grouped by object ID
	//typedef std::map<int, PointList> tracks_t;
	//tracks_t sample_tracks;
	//tracks_t found_tracks;

	float det_metric_same;
	float det_metric_good;
	float det_metric_bad;
	float det_followers_th;

	bool read_points(const std::string &path, bool to_gt, bool want_tracks);
	bool read_points(const std::string &path, PointList &all_points/*, tracks_t &tracks*/, bool want_tracks);
	void clear_points(bool gt);

	void skip_frame(int frame_num);
	void add_detections(const std::vector<ResultDetection> &objects,
		int frame_num, const std::string &frame_name = "");
	void add_tracks(const std::vector<ResultTarget> &objects, int frame_num);

	void collect_det_point_stat(int obj_type, DetectorStat &stat,
		std::vector<GroundTruthPoint*> &gt, std::vector<GroundTruthPoint*> &res);

	bool collect_detector_stat(int obj_type, DetectorStat &stat);
	void collect_detector_stat_movie(int obj_type, DetectorStat &stat);
	void collect_detector_stat_images(int obj_type, DetectorStat &stat);
	bool collect_tracker_stat(int max_frame = -1);
	void save_detector_stat(const std::string &path, int obj_type);
	void save_tracker_stat(const std::string &path);
	void save_time_stat(const std::string &path, double execution_time, int frames);

	void write_stat(const Task &task, double execution_time, int frames);
};

}  // namespace ground_truth

#endif  // AIFIL_GT_UTILS_H
