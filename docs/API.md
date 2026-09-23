# sipi-sparam-core 公共接口

## impedance

- `parallel_rc_impedance(frequency_hz, resistance_ohm, capacitance_farad=0.0)`
- `has_zero_impedance(network)`
- `replace_zero_impedance(network, z0)`
- `enforce_nonzero_z0(network, filepath)`

## renormalization

- `BatchRenormalizationCase`
- `count_batch_cases(...)`
- `iter_batch_cases(...)`
- `validate_family_ports(...)`
- `renormalize_family_case(...)`
- `build_case_filename(...)`
- `write_touchstone_with_z0(...)`

## topology

- `ChannelInfo`
- `TopologyReport`
- `DEFAULT_CLIFF_DB`
- `default_cliff_db(metric)`
- `detect_topology(...)`
- `format_report(...)`

`ChannelInfo.tx` 表示拓扑 hub，不代表真实 driver。

## ports

- `parse_port_input(...)`
- `line_port_pairs(...)`

## transfer

- `voltage_transfer(...)`
- `network_voltage_transfer(...)`
- `magnitude_db(...)`
- `voltage_crosstalk_sum_db(...)`

## time_response

- `linear_interpolate_extrapolate(...)`
- `resample_transfer_function(...)`
- `impulse_response_from_transfer(...)`
- `trapezoidal_pulse(...)`
- `fft_convolve_prefix(...)`
- `pulse_response_from_transfer(...)`
- `find_ui_center(...)`

## snr

- `MAIN_CURSOR_METHODS`
- `WindowPeak`
- `find_main_cursor(...)`
- `make_cursor_times(...)`
- `find_window_peak(...)`
- `combine_noise(...)`

## touchstone_io

- `parse_powersi_port_names(...)`
- `apply_port_name_patch(...)`
- `parse_qs_s_def(...)`
- `apply_qs_s_def_patch(...)`
