import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import apply_overrides, load_config
from utils.dataset import load_dataset
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.results_utils import (
    PhaseTimer,
    ResourceSampler,
    build_results_payload,
    gaussian_count,
    write_results_json,
)
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None, config_path=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()

        self.config = config
        self.save_dir = save_dir
        self.config_path = config_path
        self._phase_timer = PhaseTimer()
        self._resources = ResourceSampler().start()
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self._phase_timer.lap("setup")
        with self._phase_timer.phase("online"):
            self.frontend.run()
        backend_queue.put(["pause"])

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        ATE = None
        rendering_before = None
        rendering_after = None
        gaussians_after_tracking = None
        gaussians_final = None

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            gaussians_after_tracking = gaussian_count(self.gaussians)

            with self._phase_timer.phase("eval_before_opt"):
                ATE = eval_ate(
                    self.frontend.cameras,
                    self.frontend.kf_indices,
                    self.save_dir,
                    0,
                    final=True,
                    monocular=self.monocular,
                )

                rendering_before = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    iteration="before_opt",
                )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_before["mean_psnr"],
                rendering_before["mean_ssim"],
                rendering_before["mean_lpips"],
                ATE,
                FPS,
            )

            with self._phase_timer.phase("color_refinement"):
                # re-used the frontend queue to retrive the gaussians from the backend.
                while not frontend_queue.empty():
                    frontend_queue.get()
                backend_queue.put(["color_refinement"])
                while True:
                    if frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = frontend_queue.get()
                    if data[0] == "sync_backend" and frontend_queue.empty():
                        gaussians = data[1]
                        self.gaussians = gaussians
                        break

            gaussians_final = gaussian_count(self.gaussians)

            with self._phase_timer.phase("eval_after_opt"):
                rendering_after = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    iteration="after_opt",
                )
            metrics_table.add_data(
                "After",
                rendering_after["mean_psnr"],
                rendering_after["mean_ssim"],
                rendering_after["mean_lpips"],
                ATE,
                FPS,
            )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        with self._phase_timer.phase("teardown"):
            backend_queue.put(["stop"])
            backend_process.join()
            Log("Backend stopped and joined the main thread")
            if self.use_gui:
                q_main2vis.put(gui_utils.GaussianPacket(finish=True))
                gui_process.join()
                Log("GUI Stopped and joined the main thread")

        self._resources.stop()
        self._write_results(
            timer=self._phase_timer,
            total_s=self._phase_timer.total(),
            fps=FPS,
            n_frames=N_frames,
            ate=ATE,
            rendering_before=rendering_before,
            rendering_after=rendering_after,
            gaussians_after_tracking=gaussians_after_tracking,
            gaussians_final=gaussians_final,
        )

    def _write_results(
        self,
        timer,
        total_s,
        fps,
        n_frames,
        ate,
        rendering_before,
        rendering_after,
        gaussians_after_tracking,
        gaussians_final,
    ):
        """Assemble and write ``results.json`` into the run directory."""
        if not self.save_dir:
            return

        save_dir = self.save_dir
        durations = dict(timer.durations)

        # Fall back to the frontend's synced model when eval mode is off.
        if gaussians_after_tracking is None:
            gaussians_after_tracking = gaussian_count(
                getattr(self.frontend, "gaussians", None)
            )
        if gaussians_final is None:
            gaussians_final = gaussian_count(getattr(self.frontend, "gaussians", None))

        payload = build_results_payload(
            save_dir=save_dir,
            config_path=self.config_path,
            config=self.config,
            eval_rendering=self.eval_rendering,
            durations=durations,
            total_s=total_s,
            fps=fps,
            n_frames=n_frames,
            num_keyframes=len(self.frontend.kf_indices),
            ate=ate,
            rendering_before=rendering_before,
            rendering_after=rendering_after,
            gaussians_after_tracking=gaussians_after_tracking,
            gaussians_final=gaussians_final,
            memory=self._resources.summary(),
        )

        path = write_results_json(save_dir, payload)
        Log("Wrote results to " + path, tag="Eval")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override config entries with dotted keys, "
            "e.g. --set Training.gaussian_th=0.25"
        ),
    )

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=False")
        config["Results"]["use_wandb"] = False

    if args.set:
        apply_overrides(config, args.set)
        Log("Applied config overrides: " + ", ".join(args.set))

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    slam = SLAM(config, save_dir=save_dir, config_path=args.config)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
