# Fast-Approximate
Artifact Hackathon 

clean data in: Result/datasets


Data from here:
https://cernbox.cern.ch/s/yVioydShmcGKkto


| Parameter | Value | Data field used |
|---|---|---|
| `BCM_GUN` | float, **nC** | target |
| `clex_laser_power`, `lab_laser_power` | float | |
| `laser_scope` | dict: `signal` (2002,) V, `times` (2002,) ns, `saturated`, `acquired`, gain/offset | `signal` |
| `BTV0125` | dict: `image2D` (1216, 1936) int16, `isGoodExposure`, `cycleStampGlob`, `imageTimeStamp`, `acqImageNumberDiffFromLast`, calibrations, projections | `image2D` |
| `PGIA`, `PGRA`, `PGIP`, `PKR15A`, `PKI15P` | array (1, 960), arbitrary units | the array |
| `PGLA`, `PKI15A` | dict: `samples` (1, 960), arbitrary units (labelled mV), `hotSpotsSamples` (1, 2), timing | `samples` |
| `SNH_110`, `SNI_120`, `DHG0130` | float | |
| `TEMP1–4`, `HUMID1–4` | float | |



| Column | Meaning |
|---|---|
| `t` | trigger time, s since epoch |
| `pc_time` | acquisition PC clock |
| `<NAME>` | scalar parameters |
| `<NAME>.peak`, `.peak_pos`, `.integral`, `.baseline` | waveform features, baseline-subtracted, same as `plot_data.py` |
| `<NAME>.w00 … .w47` | the waveform averaged in 48 equal windows (float32) |
| `<NAME>.intensity`, `.cx`, `.cy`, `.sx`, `.sy` | camera: background-subtracted total intensity, centroid and rms size in px |
| `<NAME>.stamp_offset` | that parameter's stamp − trigger stamp, in s. ≈ 0 means the same shot; ≈ −1 shot spacing means the previous shot |
| `BTV0125.isGoodExposure`, `.acqImageNumberDiffFromLast` | camera quality flags |
| `BTV0125.cycleStampGlob_offset`, `.imageTimeStamp_offset` | the camera's own timing, relative to the trigger, in s |
| `laser_scope.saturated`, `.acquired`, `.vertOffset_V`, `.vertGain_VperLSB`, `.horizOffset_ns` | scope status and settings |
| `PGLA.hotspot0/1`, `PKI15A.hotspot0/1` | the device's own two hot-spot samples |


| Parameter | What it measures | Unit |
|---|---|---|
| `BCM_GUN` | beam charge at the gun (**target**) | nC |
| `PGIA` | gun RF, incident power | a.u. |
| `PGRA` | gun RF, reflected power | a.u. |
| `PGLA` | gun RF, pickup loop in the gun | a.u. |
| `PGIP` | gun RF, phase | a.u. |
| `PKI15A` | klystron MKS15, incident power | a.u. |
| `PKR15A` | klystron MKS15, reflected power | a.u. |
| `PKI15P` | klystron MKS15, phase | a.u. |
| `lab_laser_power` | laser energy, measured in the laser lab | J |
| `clex_laser_power` | laser energy, measured in the accelerator | J |
| `laser_scope` | fast photodiode on the laser pulse | V |
| `BTV0125` | camera on the laser spot at the virtual cathode | px |
| `SNH_110`, `SNI_120`, `DHG0130` | magnet currents (solenoids and corrector near the gun) | A |
| `TEMP1–4` | hall temperature | °C |
| `HUMID1–4` | hall humidity | % |
