#ifndef AIFIL_DUMP_UTILS_H
#define AIFIL_DUMP_UTILS_H

#include <core/io-structures.h>
#include <core/raw-structures.h>
#include <classifier/classifier.h>
//#include <tracker/mtw-tracker.h>

#include "imgproc/mat-cache.h"

#include <stdint.h>
#include <string>
#include <vector>

using anfisa::ResultDetection;
using anfisa::DetectorZoneParams;

namespace ground_truth {

struct Task;
struct GroundTruthPoint;
class ImageCooker;
/* ICF dump format:
 * .dump - raw data: samples_written samples of tile_w * tile_h size, row-major
 * .desc - dump description
 */

struct DumpStatICF : anfisa::ClassifyWindow
{
	bool integrated;
	int channels;
	std::string ch_order;
	int depth;

	int samples_written;

	DumpStatICF() : integrated(true), channels(0), depth(0), samples_written(0) {}
	std::string key() const;
	std::string to_string() const;
};

/* Detector's results dump format:
 * ts fps frame_num
 * detection_count
 * for each detection:
 *  x y w h confidence id
 *  fingerprint_count
 *  fingerprint0 ... fingerprint_n
 */

struct DumpDetectionFrame
{
	uint64_t ts;
	double fps;
	int frame_num;
	std::vector<ResultDetection> detections;
	bool load_one_detection(FILE *file, ResultDetection &me);
	void save_one_detection(FILE *file, const ResultDetection &me);

	bool load(FILE *file);
	void save(FILE *file);
};

#ifdef NEED_DUMPER
struct Dumper
{
	Dumper(const Task *task, ImageCooker *cooker = 0);
	~Dumper();

	// TODO: replace to matcache
	ImageCooker *cooker;
	const Task *task;
	std::map<std::string, DumpStatICF> dump_stat_icf;
	FILE *file_detections;
	int detections_num;

	void dump(const std::vector<GroundTruthPoint*> &vec, int frame_n);
	void dump(const std::vector<ResultDetection> &vec, int frame_n);

	bool load_detections(std::vector<DumpDetectionFrame> &dst, std::string filename = "");

	//internal
	bool dump_tiles_icf_from_rgb(const std::vector<GroundTruthPoint*> &vec, int frame_n);
	void dump_tile_icf(int frame_n, int obj_ind, MatCache &cache,
		const DumpStatICF &tile_desc, int transform_idx = 0);
	void finalize_icf();
};
#endif

}  // namespace ground_truth

#endif  // AIFIL_DUMP_UTILS_H
