import rawpy


raw = rawpy.imread('img_data/groundtruth.ARW')
gt_rgb = raw.postprocess(use_web_gain=True)