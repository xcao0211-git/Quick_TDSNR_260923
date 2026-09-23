import numpy as np
import pytest
import skrf as rf

from sipi_sparam_core.transfer import (
    magnitude_db,
    network_voltage_transfer,
    voltage_crosstalk_sum_db,
    voltage_transfer,
)


def test_voltage_transfer_equal_impedance_includes_matched_source_divider():
    result = voltage_transfer(np.array([1.0 + 0j]), 50.0, 50.0)

    np.testing.assert_allclose(result, [0.5 + 0j])
    np.testing.assert_allclose(magnitude_db(result), [-6.020599913279624])


def test_voltage_transfer_applies_impedance_ratio():
    result = voltage_transfer(np.array([1.0 + 0j]), 25.0, 100.0)

    np.testing.assert_allclose(result, [1.0 + 0j])


def test_voltage_crosstalk_sum_is_power_sum():
    aggressors = np.array([[0.5 + 0j], [0.5j]])

    result = voltage_crosstalk_sum_db(aggressors, axis=0)

    np.testing.assert_allclose(result, [-3.010299956639812])


def test_voltage_transfer_rejects_zero_reference_impedance():
    with np.testing.assert_raises_regex(ValueError, "源端参考阻抗"):
        voltage_transfer(1.0, 0.0, 50.0)


def test_network_voltage_transfer_uses_frequency_dependent_complex_zref():
    frequency = rf.Frequency(1, 2, 2, "GHz")
    s = np.zeros((2, 2, 2), dtype=complex)
    s[:, 1, 0] = [0.8 + 0.1j, 0.7 - 0.2j]
    z0 = np.array(
        [[34.0 - 2.0j, 60.0 - 8.0j], [31.0 - 5.0j, 48.0 - 15.0j]]
    )
    network = rf.Network(
        frequency=frequency, s=s, z0=z0, s_def="traveling"
    )

    result = network_voltage_transfer(network, rx_port=2, tx_port=1)

    expected = 0.5 * s[:, 1, 0] * np.sqrt(z0[:, 1] / z0[:, 0])
    np.testing.assert_allclose(result, expected)


def test_network_voltage_transfer_rejects_complex_power_wave_input():
    frequency = rf.Frequency(1, 2, 2, "GHz")
    network = rf.Network(
        frequency=frequency,
        s=np.zeros((2, 2, 2), dtype=complex),
        z0=np.array([[34.0 - 2.0j, 60.0], [32.0 - 4.0j, 60.0]]),
        s_def="power",
    )

    with pytest.raises(ValueError, match="traveling-wave"):
        network_voltage_transfer(network, rx_port=2, tx_port=1)


def test_network_voltage_transfer_allows_real_zref_legacy_network():
    frequency = rf.Frequency(1, 2, 2, "GHz")
    s = np.zeros((2, 2, 2), dtype=complex)
    s[:, 1, 0] = 1.0
    network = rf.Network(
        frequency=frequency,
        s=s,
        z0=np.array([[34.0, 60.0], [34.0, 60.0]]),
    )

    result = network_voltage_transfer(network, rx_port=2, tx_port=1)

    np.testing.assert_allclose(result, 0.5 * np.sqrt(60.0 / 34.0))
