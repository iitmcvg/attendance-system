echo $(date -u) MASK Tile 104001002F3A8700 starting >> tilelog.txt
CUDA_VISIBLE_DEVICES='0' python object_detection/inference/data_infer_batch.py \
--input_path="/media/ssd1/data/104001002F3A8700_jpg.tif_tiles/19/" \
--inference_graph="/media/ssd1/sat_data_models/mask_rcnn/large/export_batch_low_proposals/frozen_inference_graph.pb" \
--output_path="/media/ssd1/tile_output_mask_rcnn/104001002F3A8700" \
--shard_path="/media/ssd1/tile_output_mask_rcnn/104001002F3A8700" \
--batch_size=16 \
--shard=0 \
--shard_size=2 \
--vis_path="/media/ssd1/tile_output_mask_rcnn/vis/104001002F3A8700" \
--restore_from_json=False

CUDA_VISIBLE_DEVICES='3' python object_detection/inference/data_infer_batch.py \
--input_path="/media/ssd1/data/104001002F3A8700_jpg.tif_tiles/19/" \
--inference_graph="/media/ssd1/sat_data_models/mask_rcnn/large/export_batch_low_proposals/frozen_inference_graph.pb" \
--output_path="/media/ssd1/tile_output_mask_rcnn/104001002F3A8700" \
--shard_path="/media/ssd1/tile_output_mask_rcnn/104001002F3A8700" \
--batch_size=16 \
--shard=1 \
--shard_size=2 \
--vis_path="/media/ssd1/tile_output_mask_rcnn/vis/104001002F3A8700" \
--restore_from_json=False
