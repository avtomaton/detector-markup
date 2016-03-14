#include "dump-utils.h"

#include <common/errutils.h>
#include <common/stringutils.h>
#include <common/logging.h>


#include "gt-utils.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <memory>

namespace ground_truth {

std::string DumpStatICF::key() const
{
	return aifil::stdprintf("%dx%d-%d-%d-%d-%d",
		obj_w, obj_h, margin_top, margin_right, margin_bottom, margin_left);
}

std::string DumpStatICF::to_string() const
{
	std::string res;
	res += aifil::stdprintf("width          %d\n", tile_w);
	res += aifil::stdprintf("height         %d\n", tile_h);
	res += aifil::stdprintf("margin_top     %d\n", margin_top);
	res += aifil::stdprintf("margin_right   %d\n", margin_right);
	res += aifil::stdprintf("margin_bottom  %d\n", margin_bottom);
	res += aifil::stdprintf("margin_left    %d\n", margin_left);
	res += aifil::stdprintf("channels       %d ", channels) + ch_order + "\n";
	res += aifil::stdprintf("integrated     %d\n", integrated);
	res += aifil::stdprintf("depth          %d\n", depth);
	res += aifil::stdprintf("samples        %d\n", samples_written);
	return res;
}

bool DumpDetectionFrame::load_one_detection(FILE *file, ResultDetection &me)
{
	af_assert(file);
	int res = fscanf(file, "%f %f %f %f %f %d %d",
		&me.center_x, &me.center_y, &me.width, &me.height, &me.confidence, &me.id, &me.type);
	if (res != 7)
		return false;

	int count = 0;
	res = fscanf(file, "%d", &count);
	if (res != 1)
		return false;
	if (count > 0)
		me.fingerprint.resize(count);
	for (int d = 0; d < count; ++d)
	{
		res = fscanf(file, STDIO_U64F, &me.fingerprint[d]);
		if (res != 1)
			return false;
	}

	return true;
}

void DumpDetectionFrame::save_one_detection(FILE *file, const ResultDetection &me)
{
	af_assert(file);
	int res = fprintf(file, "%f %f %f %f %f %d %d\n",
		me.center_x, me.center_y, me.width, me.height, me.confidence, me.id, me.type);

	res = fprintf(file, "%d\n", int(me.fingerprint.size()));
	for (int d = 0; d < (int)me.fingerprint.size(); ++d)
	{
		res = fprintf(file, STDIO_U64F, me.fingerprint[d]);
		res = fprintf(file, " ");
	}
	res = fprintf(file, "\n");
}

bool DumpDetectionFrame::load(FILE *file)
{
	af_assert(file);
	int res = fscanf(file, STDIO_U64F, &ts);
	if (res != 1)
		return false;
	res = fscanf(file, "%le %d", &fps, &frame_num);
	if (res != 2)
		return false;
	int count = 0;
	res = fscanf(file, "%d", &count);
	if (res != 1)
		return false;
	if (count > 0)
		detections.resize(count);
	for (int d = 0; d < count; ++d)
	{
		if (!load_one_detection(file, detections[d]))
			return false;
	}
	return true;
}

void DumpDetectionFrame::save(FILE *file)
{
	af_assert(file);
	int res = fprintf(file, STDIO_U64F, ts);
	res = fprintf(file, " %e %d\n", fps, frame_num);
	res = fprintf(file, "%d\n", int(detections.size()));
	for (size_t d = 0; d < detections.size(); ++d)
		save_one_detection(file, detections[d]);
}

#ifdef NEED_DUMPER
Dumper::Dumper(const Task *task_, ImageCooker *cooker_)
	: cooker(cooker_), task(task_), file_detections(0), detections_num(0)
{
}

Dumper::~Dumper()
{
	if (task->action != "dump")
		return;

	if (!dump_stat_icf.empty())
		finalize_icf();

	if (file_detections)
	{
		fclose(file_detections);
		aifil::log_state("DUMP: %d detections saved in '%s/%s.det_dump'", detections_num,
			task->dump_path.c_str(), task->my_name.c_str());
	}
}

void Dumper::dump(const std::vector<GroundTruthPoint*> &vec, int frame_n)
{
	if (task->action != "dump")
		return;

	if (!task->dump_tile_params.empty() && !vec.empty())
		dump_tiles_icf_from_rgb(vec, frame_n);
}

void Dumper::dump(const std::vector<ResultDetection> &vec, int frame_n)
{
	if (task->action != "dump")
		return;

	if (vec.empty())
		return;

	if (!file_detections)
	{
		std::string fname = task->dump_path + "/" + task->my_name + gt_det_dump_ext;
		file_detections = fopen(fname.c_str(), "wb");
	}

	if (!file_detections)
	{
		aifil::log_warning("DUMP: cannot open dump file for writing\n");
		return;
	}

	DumpDetectionFrame fr;
	fr.ts = cooker->real_timestamp;
	fr.fps = cooker->current_fps;
	fr.frame_num = frame_n;
	fr.detections = vec;
	fr.save(file_detections);
	++detections_num;
}

bool Dumper::load_detections(std::vector<DumpDetectionFrame> &dst, std::string filename)
{
	if (filename.empty())
		filename = task->dump_path + "/" + task->my_name + gt_det_dump_ext;

	FILE *file = fopen(filename.c_str(), "rb");
	if (!file)
		return false;

	bool stop = false;
	while (!stop)
	{
		DumpDetectionFrame cur;
		if (cur.load(file))
			dst.push_back(cur);
		else
			break;
	}
	fclose(file);

	if (dst.empty())
		aifil::log_warning("DUMP: cannot load detections");
	else
		aifil::log_state("DUMP: loaded %d detections", dst.size());

	return !dst.empty();
}

void Dumper::dump_tile_icf(int frame_n, int obj_ind,
	MatCache &cache, const DumpStatICF &tile_desc, int transform_idx)
{
	std::string dump_name = aifil::stdprintf("%s/%s-%s-%s",
		task->dump_path.c_str(), task->dump_prefix.c_str(),
		tile_desc.key().c_str(), task->my_name.c_str());

	cache.invalidate();
	cache.img_rgb_ready = true;
	if (task->dump_images)
		cv::imwrite(aifil::stdprintf("%s/%s-%d-%d-%05i-%d-%d.png",
			task->export_path.c_str(), task->dump_prefix.c_str(),
			tile_desc.obj_w, tile_desc.obj_h, frame_n, obj_ind, transform_idx), cache.img_rgb);

	if (task->dump_type == "icf_good")
		cache.icf_create();
	else if (task->dump_type == "icf_canonical")
		cache.icf_create("canonical");
	else if (task->dump_type == "icf_dssl")
		cache.icf_create("dssl");
	else
	{
		aifil::log_warning("DUMP: incorrect ICF type requested");
		return;
	}

	FILE* dump_file = 0;
	af_assert(cache.icf_integral_ready);
	if (dump_stat_icf.find(dump_name) == dump_stat_icf.end())
	{
		dump_stat_icf[dump_name] = tile_desc;
		dump_stat_icf[dump_name].tile_w = cache.img_rgb.cols;
		dump_stat_icf[dump_name].tile_h = cache.img_rgb.rows;
		dump_stat_icf[dump_name].integrated = true;
		dump_stat_icf[dump_name].channels = 1 + cache.icf_hog_bins + cache.icf_color_bands;
		dump_stat_icf[dump_name].ch_order =
			aifil::stdprintf("gr hog%d val%d", cache.icf_hog_bins, cache.icf_color_bands);
		dump_stat_icf[dump_name].depth = cache.icf_integral.depth();
		dump_file = fopen((dump_name + gt_icf_dump_ext).c_str(), "wb");
	}
	else
	{
		af_assert(dump_stat_icf[dump_name].tile_w == cache.img_rgb.cols);
		af_assert(dump_stat_icf[dump_name].tile_h == cache.img_rgb.rows);
		af_assert(dump_stat_icf[dump_name].margin_top == tile_desc.margin_top);
		af_assert(dump_stat_icf[dump_name].margin_right == tile_desc.margin_right);
		af_assert(dump_stat_icf[dump_name].margin_bottom == tile_desc.margin_bottom);
		af_assert(dump_stat_icf[dump_name].margin_left == tile_desc.margin_left);
		af_assert(dump_stat_icf[dump_name].depth == cache.icf_integral.depth());
		dump_file = fopen((dump_name + gt_icf_dump_ext).c_str(), "ab");
	}

	if (!dump_file)
	{
		aifil::log_warning("DUMP: can't open file");
		return;
	}

	const cv::Mat &tile = cache.icf_integral;
	if (fwrite(tile.data, tile.elemSize(), tile.rows * tile.cols, dump_file) == tile.rows * tile.cols)
		dump_stat_icf[dump_name].samples_written++;
	else
		aifil::log_warning("DUMP: failed writing %d-%d-%d", frame_n, obj_ind, tile_desc.obj_w);

	fclose(dump_file);
}

bool Dumper::dump_tiles_icf_from_rgb(const std::vector<GroundTruthPoint*> &vec, int frame_n)
{
	af_assert(cooker && "need cooker for ICF dump");
	bool keep_pic_aspect_ratio = false;
	std::shared_ptr<MatCache> orig_scale = cooker->scale_find(cooker->orig_w, cooker->orig_h, true);
	const cv::Mat* img_full = cooker->get_img_rgb(cooker->orig_w, cooker->orig_h);
	if (!img_full)
	{
		aifil::log_warning("DUMP: can't capture RGB frame");
		return false;
	}

	for (int obj_ind = 0; obj_ind < (int)vec.size(); ++obj_ind)
	{
		const GroundTruthPoint *obj = vec[obj_ind];
		if (!obj->object_type)
		{
			aifil::log_warning("DUMP: dummy negatives dump, nothing to do!");
			continue;
		}
		else if (obj->object_type != task->object_type) //other detector's object
			continue;

		for (std::list<DumpStatICF>::const_iterator tile_sc_it = task->dump_tile_params.begin();
			tile_sc_it != task->dump_tile_params.end(); ++tile_sc_it)
		{
			const DumpStatICF& tile_sc = *tile_sc_it;
			double scale_x = (obj->width / 100.0 * cooker->orig_w) / double(tile_sc.obj_w);
			double scale_y = (obj->height / 100.0 * cooker->orig_h) / double(tile_sc.obj_h);
			if (scale_x < 1.0 || scale_y < 1.0)
			{
				aifil::log_state("DUMP: upscale at frame=%i, scale_w==%0.2lf, scale_h==%0.2lf",
					frame_n, scale_x, scale_y);
				continue;
			}
			double ratio = scale_y / scale_x;
			if (task->dump_pedantic_size && (ratio >= 1.2 || ratio <= 0.8))
			{
				aifil::log_state("DUMP: aspect ratio frame=%i, scale_w==%0.2lf, scale_h==%0.2lf",
					frame_n, scale_x, scale_y);
				continue;
			}

			double gt_obj_w = obj->width;
			double gt_obj_h = obj->height;
			if (keep_pic_aspect_ratio)
			{
				//create minimal bounding rect
				if (ratio > 1.0)
					gt_obj_h = obj->height * ratio;
				else
					gt_obj_w = obj->width / ratio;
			}

			int crop_left = int((obj->center_x - gt_obj_w / 2) / 100.0 * cooker->orig_w -
				tile_sc.margin_left * scale_x);
			int crop_up = int((obj->center_y - gt_obj_h / 2) / 100.0 * cooker->orig_h -
				tile_sc.margin_top * scale_y);
			int crop_right = int((obj->center_x + gt_obj_w / 2) / 100.0 * cooker->orig_w +
				tile_sc.margin_right * scale_x);
			int crop_down = int((obj->center_y + gt_obj_h / 2) / 100.0 * cooker->orig_h +
				tile_sc.margin_bottom * scale_y);

			if (crop_left < 0 || crop_up < 0 ||
				crop_right >= img_full->cols || crop_down >= img_full->rows)
			{
				aifil::log_state("DUMP: offscreen frame=%i, tile %dx%d", frame_n, tile_sc.obj_w, tile_sc.obj_h);
				continue;
			}

			MatCache cache(tile_sc.tile_w, tile_sc.tile_h);
			cache.img_rgb = cv::Mat(tile_sc.tile_h, tile_sc.tile_w, CV_8UC3);
			const cv::Mat cropped = cv::Mat(*img_full,
				cv::Rect(crop_left, crop_up, crop_right - crop_left, crop_down - crop_up));
			cv::resize(cropped, cache.img_rgb, cache.img_rgb.size());
			cache.img_rgb_ready = true;

			dump_tile_icf(frame_n, obj_ind, cache, tile_sc, 0);

			//flip horizontally
			cv::flip(cache.img_rgb, cache.img_rgb, 1);
			dump_tile_icf(frame_n, obj_ind, cache, tile_sc, 1);

			//TODO: maybe small shift and affine transformations?

		} //for each tile size
	} //for each object from gt

	return true;
}

void Dumper::finalize_icf()
{
	for (std::map<std::string, DumpStatICF>::const_iterator it = dump_stat_icf.begin();
		 it != dump_stat_icf.end(); ++it)
	{
		const DumpStatICF &cur = it->second;
		if (cur.samples_written == 0)
			continue;

		FILE *desc = fopen((it->first + gt_icf_desc_ext).c_str(), "wb");
		if (!desc)
		{
			aifil::log_warning("DUMP: can't save description for '%s'!", it->first.c_str());
			continue;
		}
		fprintf(desc, "%s", cur.to_string().c_str());
		fclose(desc);
		aifil::log_state("DUMP: saved %d samples to '%s'", cur.samples_written, it->first.c_str());
	}
}
#endif
//end NEED_DUMPER

}  // namespace ground_truth
