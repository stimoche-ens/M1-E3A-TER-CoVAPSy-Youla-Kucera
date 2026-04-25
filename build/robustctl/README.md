# robustctl

This folder is the robust-control bridge between the identified linear model,
the Youla-Kucera training data, and the Webots runtime controller.

## Structure

Root configuration lives in `conf/`. The local `conf.py` file is a compatibility
shim for existing scripts that still do `import conf`.

`linear_model.py` loads the ARX parameters from `build/linmodel` and builds the
discrete MIMO state-space plant `H`.

The runtime convention is `eps = D_nom - lidar`. New `linmodel` artifacts store
that convention in metadata. Older notebook JSON files did not, so
`linear_model.py` treats missing metadata as legacy `lidar - D_nom` and converts
the input-side coefficients on load.

`kcontroller.py` synthesizes the static robust baseline controller `K0`, exports
controller artifacts, and provides the runtime `StaticOutputFeedbackController`
used by Webots.

`nn_runtime.py` loads the `MyLSTM` weights trained on `UQYQ` and exposes a
bounded neural forecast of future `u_q` for runtime inference.

`io_pipeline.py` generates `u_q/y_q` CSV files for future Youla-Kucera `Q`
training. It reuses the `linmodel` column normalizer, so real `lidar[...]`
files and simulation `lidar_0...lidar_359` files follow the same signed-angle
convention.

`build_controller.py` is the main artifact builder:

```bash
python3 build/robustctl/build_controller.py --deploy
```

This writes `build/robustctl/generated/robust_controller.json` and deploys a
timestamped controller artifact under `build/robustctl/generated/`, updates
`conf/current_artifacts.json`, and deploys a stable Webots copy to
`inference/webots_sim/controllers/controller_jaune/robust_controller.json`.

`identify_inputs_outputs.py` remains as the compatibility CLI for generating
`u_q/y_q` files.
