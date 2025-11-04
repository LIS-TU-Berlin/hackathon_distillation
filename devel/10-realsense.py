import pyrealsense2 as rs
import numpy as np
import cv2
import torch

from hackathon_distillation.networks.da_encoder import DaEncoder

def build_filters(
    decimation_mag=2,
    spatial_mag=2,           # 1..5 (kernel size). Larger -> stronger smoothing.
    spatial_alpha=0.5,       # 0..1  (0=aggressive smoothing, 1=preserve edges)
    spatial_delta=20,        # 1..50 (edge threshold; higher preserves more edges)
    spatial_holes_fill=3,    # 0..5  (in-pixel hole fill within spatial filter)
    temporal_alpha=0.4,      # 0..1  (history weight; higher = steadier, slower)
    temporal_delta=20,       # 1..100(motion threshold)
    temporal_persistency=3,  # 0..8  (reanimation strength; 3~4 is sensible)
    hole_filling_mode=2      # 0=disabled, 1=fill from left-right, 2=closest
):
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, decimation_mag)

    depth_to_disp = rs.disparity_transform(True)
    disp_to_depth = rs.disparity_transform(False)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, spatial_mag)
    spatial.set_option(rs.option.filter_smooth_alpha, spatial_alpha)
    spatial.set_option(rs.option.filter_smooth_delta, spatial_delta)
    spatial.set_option(rs.option.holes_fill, spatial_holes_fill)

    temporal = rs.temporary_filter() if hasattr(rs, "temporary_filter") else rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, temporal_alpha)
    temporal.set_option(rs.option.filter_smooth_delta, temporal_delta)
    # Persistency controls how aggressively we "reanimate" missing pixels using history
    if hasattr(rs.option, "persistency_index"):
        temporal.set_option(rs.option.persistency_index, temporal_persistency)

    hole_filling = rs.hole_filling_filter()
    if hasattr(rs.option, "holes_fill"):
        try:
            hole_filling.set_option(rs.option.holes_fill, hole_filling_mode)
        except Exception:
            pass  # older SDKs set mode by constructor only

    return {
        "decimation": decimation,
        "depth_to_disp": depth_to_disp,
        "spatial": spatial,
        "temporal": temporal,
        "disp_to_depth": disp_to_depth,
        "hole_filling": hole_filling
    }

def postprocess_depth(depth_frame, filters):
    f = depth_frame
    f = filters["decimation"].process(f)
    f = filters["depth_to_disp"].process(f)
    f = filters["spatial"].process(f)
    f = filters["temporal"].process(f)
    f = filters["disp_to_depth"].process(f)
    f = filters["hole_filling"].process(f)
    return f

def main(live=True, bag_path=None, show=True):
    """
    live=True  -> stream from camera
    live=False -> play a .bag recording via bag_path
    """
    import robotic as ry
    import hackathon_distillation as hack

    #S = hack.Scene()
    #q0 = S.C.getJointState()
    #bot = ry.BotOp(C=S.C, useRealRobot=live)
    vis = hack.DataPlayer(np.zeros((340, 360, 3)), np.zeros((340, 360, 3)))

    pipeline = rs.pipeline()
    config = rs.config()

    if live:
        # Depth + color so we can align and visualize
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    else:
        if not bag_path:
            raise ValueError("bag_path must be provided when live=False")
        config.enable_device_from_file(bag_path, repeat_playback=True)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        if depth_sensor.supports(rs.option.laser_power):
            # Use ~70–100% indoors; tweak for your scene
            rng = depth_sensor.get_option_range(rs.option.laser_power)
            depth_sensor.set_option(rs.option.laser_power, min(rng.max, max(rng.min, 200.0)))
    except Exception:
        pass

    # Filters
    filters = build_filters(
        decimation_mag=2,
        spatial_mag=2,
        spatial_alpha=0.8,
        spatial_delta=20,
        spatial_holes_fill=3,
        temporal_alpha=0.4,
        temporal_delta=20,
        temporal_persistency=3,
        hole_filling_mode=2
    )

    colorizer = rs.colorizer()  # for quick visualization

    da_model = DaEncoder()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            frames = align.process(frames)

            depth = frames.get_depth_frame()
            color = frames.get_color_frame()
            if not depth:
                continue

            depth_pp = postprocess_depth(depth, filters)

            if show:
                # Visualize raw vs processed (colorized)
                depth_col_raw = np.asanyarray(depth.get_data()).copy() / 1000
                depth_col_pp  = np.asanyarray(depth_pp.get_data()).copy() / 1000
                depth_da = np.asanyarray(color.get_data()).copy()
                # depth_da = da_model(
                #     torch.from_numpy(depth_da), 
                #     depth=torch.from_numpy(depth_col_pp)[None]
                # ).detach().cpu().numpy()[0]

                vis.update(depth_da, depth_da)
            else:
                # If you just want arrays:
                depth_meters = np.asanyarray(depth_pp.get_data()) * profile.get_device().first_depth_sensor().get_depth_scale()
                print("Processed depth shape:", depth_meters.shape)
                break

    finally:
        pipeline.stop()
        #cv2.destroyAllWindows()

if __name__ == "__main__":
    main(live=True)
