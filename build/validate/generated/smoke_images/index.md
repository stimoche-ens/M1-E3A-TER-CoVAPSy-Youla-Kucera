# Validation Report

Output directory: `build/validate/generated/smoke_images`

## linmodel
- `mean_rmse`: `1.4663083652863804`
- `mean_r2`: `-13.594897116776021`

### Figures

#### linmodel rmse heatmap

![linmodel rmse heatmap](linmodel/linmodel_rmse_heatmap.png)

#### linmodel r2 heatmap

![linmodel r2 heatmap](linmodel/linmodel_r2_heatmap.png)

#### linmodel predictions

![linmodel predictions](linmodel/linmodel_predictions.png)

#### linmodel a coeffs

![linmodel a coeffs](linmodel/linmodel_a_coeffs.png)

#### linmodel b v

![linmodel b v](linmodel/linmodel_b_v.png)

#### linmodel b delta

![linmodel b delta](linmodel/linmodel_b_delta.png)


## robustctl
- `stable`: `True`
- `closed_peak`: `3.342990958304604`

### Figures

#### robustctl K heatmap

![robustctl K heatmap](robustctl/robustctl_K_heatmap.png)

#### robustctl frequency response

![robustctl frequency response](robustctl/robustctl_frequency_response.png)


## nntrain
- `status`: `ok`
- `first_step_rmse_mean`: `30.957582625527866`
- `all_horizon_rmse_mean`: `22.180842456284896`

### Figures

#### nn rmse by output

![nn rmse by output](nntrain/nn_rmse_by_output.png)

#### nn horizon rmse

![nn horizon rmse](nntrain/nn_horizon_rmse.png)

#### nn example prediction

![nn example prediction](nntrain/nn_example_prediction.png)

