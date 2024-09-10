#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import sigmf
from scipy.io import wavfile
from scipy.signal import butter, hilbert, resample_poly, sosfilt


def ease_in_out_quad(start_value, stop_value, duration, samp_rate):
    """Generate an array of values that ease in and out quadratically from start_value to stop_value over duration seconds."""
    change = stop_value - start_value
    total_samples = int(duration * samp_rate)
    t = np.linspace(0, 1, total_samples, endpoint=False)
    values = np.where(
        t < 0.5,
        start_value + change * 2 * t * t,
        start_value + change * (-1 + (4 - 2 * t) * t),
    )
    return values.astype(np.float32)


def generate_swept_sawtooth_jammer(
    samp_rate,
    duration_s,
    sweep_period_s,
    sweep_bandwidth_hz,
    noise_bandwidth_hz,
    reverse=False,
):
    """
    Jammer should generate a bandwidth_hz noise signal that sweeps in a sawtooth pattern over sweep_bandwidth_hz every sweep_period_s seconds.
    """
    noise = np.random.randn(int(duration_s * samp_rate)) + 1j * np.random.randn(
        int(duration_s * samp_rate)
    )
    # fade in & out for 20% of duration
    fade_samples = int(0.2 * duration_s * samp_rate)
    fade_in = ease_in_out_quad(0, 1, fade_samples / samp_rate, samp_rate)
    noise[:fade_samples] *= fade_in
    fade_out = ease_in_out_quad(1, 0, fade_samples / samp_rate, samp_rate)
    noise[-fade_samples:] *= fade_out

    # apply a low-pass filter to limit the noise bandwidth
    sos = butter(10, noise_bandwidth_hz / (samp_rate / 2), btype="low", output="sos")
    filtered_noise = sosfilt(sos, noise).astype(np.complex64)
    # multiply filtered_noise by a sawtooth wave that sweeps over sweep_bandwidth_hz every sweep_period_s seconds
    t_sawtooth = np.arange(len(filtered_noise)) / samp_rate
    sawtooth_wave = 0.5 * (
        2 * (t_sawtooth / sweep_period_s % 1) - 1
    )  # sawtooth wave from -0.5 to 0.5
    if reverse:
        sawtooth_wave = -sawtooth_wave
    instantaneous_freq = (sawtooth_wave + 0.5) * sweep_bandwidth_hz
    phasor = np.exp(1j * 2 * np.pi * np.cumsum(instantaneous_freq) / samp_rate)
    return (filtered_noise * phasor).astype(np.complex64)


def rotate(samples, samp_rate, frequency):
    """Rotate the signal by multiplying by a complex exponential at the given frequency."""
    t = np.arange(len(samples)) / samp_rate
    return samples * np.exp(1j * 2 * np.pi * frequency * t)


if __name__ == "__main__":
    rf_freq_hz = 443e6
    samp_rate = 480_000
    final = np.zeros(20 * samp_rate, dtype=np.complex64)
    # create the metadata
    meta = sigmf.SigMFFile(
        global_info={
            sigmf.SigMFFile.DATATYPE_KEY: sigmf.utils.get_data_type_str(final),
            sigmf.SigMFFile.SAMPLE_RATE_KEY: samp_rate,
            sigmf.SigMFFile.AUTHOR_KEY: "Kyle Logue (K6OF) & Cristina Logue (KM6YOU)",
            # SigMFFile.DESCRIPTION_KEY: 'Time to get the FLL working.',
        }
    )
    # create a capture key at time index 0
    meta.add_capture(
        0,
        metadata={
            sigmf.SigMFFile.FREQUENCY_KEY: rf_freq_hz,
            sigmf.SigMFFile.DATETIME_KEY: sigmf.utils.get_sigmf_iso8601_datetime_now(),
        },
    )

    if True:
        # flag 1: stargate, HD AM
        samples = np.fromfile(
            "part/stargate.cf32", dtype=np.complex64
        )  # sampled at 480 kHz
        print("dbug hdam", samples.shape, samples.dtype)
        # truncate to last 20 seconds
        samples = samples[-20 * samp_rate :]

    else:
        # flag 1: stargate, wideband AM
        wav_rate, samples = wavfile.read("stargate.wav")
        # only use left channel
        samples = samples[:, 0]
        # normalize to -1 to 1 by dividing by max int16 value
        samples = samples.astype(np.float32) / (2**15 - 1)
        print(np.max(samples), np.min(samples), "minmax")
        # upsample by 10
        samples = resample_poly(samples, 10, 1)
        # AM modulate onto a carrier (e.g., 20 kHz)
        carrier_freq = 20000  # Hz
        t = np.arange(len(samples)) / samp_rate
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        # AM modulation (DSB, no carrier suppression)
        samples = (1.0 + samples) * carrier
        print(len(samples), len(samples) / samp_rate)

    # phasor will ease between the offsets over 3 seconds
    offsets = [0, 60000, 120000, 0]
    phase = np.zeros(len(samples), dtype=np.float64)
    phase[5 * samp_rate : 8 * samp_rate] = ease_in_out_quad(
        offsets[0], offsets[1], 3, samp_rate
    )
    phase[8 * samp_rate : 10 * samp_rate] = offsets[1]  # hold at 5 KHz for 2 seconds
    phase[10 * samp_rate : 13 * samp_rate] = ease_in_out_quad(
        offsets[1], offsets[2], 3, samp_rate
    )
    phase[13 * samp_rate : 17 * samp_rate] = offsets[2]  # hold at 15 KHz
    phase[17 * samp_rate : 20 * samp_rate] = ease_in_out_quad(
        offsets[2], offsets[3], 3, samp_rate
    )
    ease_phasor = np.exp(1j * 2 * np.pi * np.cumsum(phase) / samp_rate)
    # plt.plot(np.cumsum(phase))
    # plt.show()
    # meta2 = sigmf.SigMFFile()
    final += samples * ease_phasor

    # flag decoy: platypus spectrum painter
    paint_samps = np.fromfile(
        "part/platypus.cf32", dtype=np.complex64
    )  # sampled at 48 KHz
    print("dbug paint", paint_samps.shape, paint_samps.dtype, len(paint_samps) / 48000)
    paint_samps = resample_poly(paint_samps, 4, 1)
    # pad to 20 seconds
    paint_samps = np.hstack(
        (
            paint_samps,
            np.zeros(
                max(0, 20 * samp_rate - len(paint_samps)), dtype=paint_samps.dtype
            ),
        )
    )
    # shift left 100 KHz
    paint_samps = rotate(paint_samps, samp_rate, -160e3)
    # roll to offset in time
    paint_samps = np.roll(paint_samps, round(1.4 * samp_rate))
    final += paint_samps * 0.1

    # no flag, just razzle dazzle
    paint_samps = np.fromfile(
        "part/razzle.cf32", dtype=np.complex64
    )  # sampled at 48 KHz
    print("dbug paint", paint_samps.shape, paint_samps.dtype, len(paint_samps) / 48000)
    paint_samps = resample_poly(paint_samps, 4, 1)
    # pad to 20 seconds
    paint_samps = np.hstack(
        (
            paint_samps,
            np.zeros(
                max(0, 20 * samp_rate - len(paint_samps)), dtype=paint_samps.dtype
            ),
        )
    )
    # shift left 100 KHz
    paint_samps = rotate(paint_samps, samp_rate, 160e3)
    # roll to offset in time
    paint_samps = np.roll(paint_samps, round(2 * samp_rate))
    final += paint_samps * 0.1

    # flag 5 & 6: modulate a LSB and USB signal 12.5 kHz apart using AM-USB and AM-LSB
    # resample from 44100 to 480000
    _, hl_lsb = wavfile.read("hl_14142135.wav")
    _, hl_usb = wavfile.read("hl_1732050807.wav")
    hl_lsb = hl_lsb.astype(np.float32) / (2**15 - 1)
    hl_usb = hl_usb.astype(np.float32) / (2**15 - 1)
    hl_lsb = resample_poly(hl_lsb, 1600, 147)
    hl_usb = resample_poly(hl_usb, 1600, 147)
    # modulate to AM-LSB and AM-USB
    f_center = -40e3
    t_hl = np.arange(len(hl_lsb)) / samp_rate
    # AM-USB: modulate with carrier, then keep only upper sideband (analytic signal)
    hl_usb_analytic = hilbert(hl_usb)
    am_usb = np.exp(1j * 2 * np.pi * (f_center + 6.25e3) * t_hl) * hl_usb_analytic
    # AM-LSB: modulate with carrier, then keep only lower sideband (analytic signal, conjugate)
    hl_lsb_analytic = hilbert(hl_lsb)
    am_lsb = np.exp(1j * 2 * np.pi * (f_center - 6.25e3) * t_hl) * np.conj(
        hl_lsb_analytic
    )
    # pad to length of final
    am_lsb = np.hstack(
        (am_lsb, np.zeros(max(0, len(final) - len(am_lsb)), dtype=am_lsb.dtype))
    )
    am_usb = np.hstack(
        (am_usb, np.zeros(max(0, len(final) - len(am_usb)), dtype=am_usb.dtype))
    )
    # roll to offset in time
    am_lsb = np.roll(am_lsb, int(11.42 * samp_rate))
    am_usb = np.roll(am_usb, int(11.42 * samp_rate))
    final += am_lsb * ease_phasor * 0.03
    final += am_usb * ease_phasor * 0.04
    # write annotations for each 5 kHz signal
    # This is somewhat tricky because the ease_phasor makes the signal start at a strange frequency offset
    strange_offset_hz = 48e3  # found experimentally
    meta.add_annotation(
        start_index=int(11.42 * samp_rate),
        length=int(samp_rate * 0.08),
        # length=len(hl_lsb),
        metadata={
            # each signal is 5 KHz Wide
            sigmf.SigMFFile.FLO_KEY: rf_freq_hz + strange_offset_hz - 8e3 - 4e3,
            sigmf.SigMFFile.FHI_KEY: rf_freq_hz + strange_offset_hz - 8e3 + 4e3,
            sigmf.SigMFFile.COMMENT_KEY: "Start of Flag 5",
        },
    )
    meta.add_annotation(
        start_index=int(11.42 * samp_rate),
        # length=len(hl_usb),
        length=int(samp_rate * 0.08),
        metadata={
            sigmf.SigMFFile.FLO_KEY: rf_freq_hz + strange_offset_hz + 8e3 - 4e3,
            sigmf.SigMFFile.FHI_KEY: rf_freq_hz + strange_offset_hz + 8e3 + 4e3,
            sigmf.SigMFFile.COMMENT_KEY: "Start of Flag 6",
        },
    )

    # flag 3: monstera, narrowband FM
    _, mon_samps = wavfile.read("monstera.wav")
    mon_samps = mon_samps.astype(np.float32)[:, 0] / (2**15 - 1)
    mon_samps = resample_poly(mon_samps, 10, 1)  # now we are at samp_rate
    print("dbug", mon_samps.shape, mon_samps.dtype)
    # print(len(mon_samps), len(mon_samps)/wav_rate,'monstera')
    # frequency modulate monstera with 3KHz deviation
    kf = 3000  # frequency deviation
    t_mon = np.arange(len(mon_samps)) / samp_rate
    integral_of_mon = np.cumsum(mon_samps) / samp_rate
    print("dbug", len(mon_samps), len(t_mon), len(integral_of_mon), len(final))
    fm_mon = np.exp(1j * 2 * np.pi * (100e3 * t_mon + kf * integral_of_mon))
    # amplitude: fade in 3 second, fade out 3 second
    fade_in = ease_in_out_quad(0, 1, 3, samp_rate)
    fade_out = ease_in_out_quad(1, 0, 3, samp_rate)
    fm_mon[: 3 * samp_rate] *= fade_in
    fm_mon[-3 * samp_rate :] *= fade_out
    # pad to length of shifted samples
    fm_mon = np.hstack(
        (fm_mon, np.zeros(max(0, len(final) - len(fm_mon)), dtype=fm_mon.dtype))
    )
    # roll to offset in time
    fm_mon = np.roll(fm_mon, 16 * samp_rate)

    final += fm_mon * ease_phasor * 0.1

    # flag 11: morse code along diagonal, 17wpm
    _, morse = wavfile.read("morsecode_0rq1tph4vgeebiqothm2e3loi1.wav")
    morse = (morse.astype(np.float32) - 128) / (2**7 - 1)  # pcm_u8 -> -1..1
    morse = resample_poly(morse, 48000, 1105)  # 11050 Hz -> 480 kHz
    # bandpass around the 550 Hz morse tone to kill keying clicks / splatter
    morse = morse - morse.mean()
    sos = butter(
        N=4,
        Wn=[300 / (samp_rate / 2), 800 / (samp_rate / 2)],
        btype="band",
        output="sos",
    )
    morse = sosfilt(sos, morse)
    print(len(morse), len(morse) / samp_rate, "morse")
    phase = np.linspace(240e3 - 40e3, -240e3 - 40e3, len(final))
    morse_padded = np.hstack(
        (morse, np.zeros(len(final) - len(morse), dtype=morse.dtype))
    )
    morse_padded = np.roll(morse_padded, 0.5 * samp_rate)
    # convert real audio to analytic signal (SSB) before applying the sweep
    morse_analytic = hilbert(morse_padded)
    morse_shifted = morse_analytic * np.exp(
        1j * 2 * np.pi * np.cumsum(phase) / samp_rate
    )
    final += morse_shifted * 5e-3

    # add swept sawtooth FM modulated jammer
    jammer = generate_swept_sawtooth_jammer(
        samp_rate,
        duration_s=5.5,
        sweep_period_s=0.15,
        sweep_bandwidth_hz=20000,
        noise_bandwidth_hz=3e3,
    )
    # frequency shift jammer to -200 kHz
    jammer = rotate(jammer, samp_rate, -140e3)
    # pad to length of final
    jammer = np.hstack(
        (
            np.zeros(samp_rate),
            jammer,
            np.zeros(len(final) - len(jammer) - samp_rate, dtype=jammer.dtype),
        )
    )
    # roll to offset in time
    # jammer = np.roll(jammer, 1 * samp_rate)
    final += jammer * np.sqrt(2)  # scale down
    meta.add_annotation(
        start_index=1 * samp_rate,
        length=5.5 * samp_rate,
        metadata={
            sigmf.SigMFFile.FLO_KEY: rf_freq_hz - 140e3,
            sigmf.SigMFFile.FHI_KEY: rf_freq_hz - 140e3 + 20e3,
            sigmf.SigMFFile.COMMENT_KEY: "Swept Jammer blocking Flag 2",
        },
    )

    # # flag 7 & 8: quiet M17 digital voice & metadata
    quiet_samps = np.fromfile("quiet_m17_california.cf32", dtype=np.complex64)
    print("quiet", len(quiet_samps), len(quiet_samps) / samp_rate, "m17")
    quiet_samps = rotate(quiet_samps, samp_rate, -70e3)
    final += quiet_samps[0 : len(final)] * ease_phasor * 1e-2

    # # add another swept sawtooth FM modulated jammer
    # jammer2 = generate_swept_sawtooth_jammer(samp_rate, duration_s=5, sweep_period_s=0.5, sweep_bandwidth_hz=30000, noise_bandwidth_hz=5e3, reverse=True)
    # # frequency shift jammer2 to -200 kHz
    # jammer2 = rotate(jammer2, samp_rate, -180e3)
    # # pad to length of final
    # jammer2 = np.hstack((jammer2, np.zeros(len(final) - len(jammer2), dtype=jammer2.dtype)))
    # # roll to offset in time
    # jammer2 = np.roll(jammer2, int(0.3 * samp_rate))
    # final += jammer2 * 0.01  # scale down

    # add a -90 dB noise AWGN noise floor
    noise_power = 10 ** (-90 / 10)
    noise = np.sqrt(noise_power) * (
        np.random.randn(len(final)) + 1j * np.random.randn(len(final))
    )
    final += noise

    # save to file
    final.astype(np.complex64).tofile("wavy.sigmf-data")
    meta.set_data_file("wavy.sigmf-data")

    print(meta)

    # flag 4: in a nonstandard metadata field (acrostic: first letters spell flagfourisfleebjuice)
    meta.set_global_field(
        "core:acrostic",
        "Found this recording on an old SDR. "
        "Lots of signals hiding in it. "
        "Amplitude modulation everywhere. "
        "Guessing there are flags inside. "
        "Four or maybe more. "
        "One signal sounded like a platypus. "
        "Underneath all that noise. "
        "Really strange sounds came out. "
        "I kept listening anyway. "
        "Some tones slid across the band. "
        "Faint morse code appeared later. "
        "Low and slow, it swept downward. "
        "Every sweep ended somewhere new. "
        "Eventually I found the digital modes. "
        "Bits and bytes in the metadata too. "
        "Just look closely at the JSON. "
        "Under the nonstandard fields. "
        "Interesting things hide there. "
        "Curious what else is buried. "
        "Everything you need is in this file.",
    )

    # check for mistakes & write to disk
    meta.tofile("wavy.sigmf-meta", overwrite=True)

    meta.tofile("../../public/wavy/wavy.sigmf", overwrite=True)
