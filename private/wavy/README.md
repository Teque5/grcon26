# Wavy

![preview](preview_gqrx.webp)

## Notes

2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199

- welcome 0-5s
- follow 5-8s
- what was that 8-9s
- follow 10-13

## Procedure

This was quite complex to create.
1) run `hd_am.grc` to create the AM HD-Radio Broadcast signal `stargate.cf32`
2) validate HD Radio Broadcast signal w/ `python3 ~/Source/nrsc5/support/cli.py --am --iq-input-format cs16 -r /tmp/hdam.cs16 -o out.wav -l 1 0`
3) run `m17_encoder_quiet.grc` to create `quiet_m17_california.cf32`
4) run `paint_flag.grc` to create both `platypus.cf32` and `razzle.cf32` for spectrum paint. I found `@argilo`'s method to be superior to `gr-paint`. 
5) Use the half-life VOX engine to create sounds
6) Use [morse generator](https://www.meridianoutpost.com/resources/etools/calculators/calculator-morse-code.php) to make flag 11
7) Run `shift.py` to bake the final file.

## Flags

### Flag 1

- Modulation: Wideband AM w/ Frequency Shifting
- Difficulty: Easy
- Points: 17
- Note: Welcome to GrCon message
- Solution: `flag{stargate}`

### Flag 2

- Modulation: Spectrum Paint
- Difficulty: Easy
- Points: 19
- Note: Clue in SigMF Metadata
- Solution: `flag{platypus}`

### Flag 3

- Modulation: Narrowband FM w/ Frequency Shifting
- Difficulty: Easy
- Points: 23
- Solution: `flag{monstera}`

### Flag 4

- Note: Hidden in SigMF Metadata
- Difficulty: Easy, wordplay
- Points: 7
- Solution: `flag{fleebjuice}`

### Flag 5

- Modulation: AM LSB w/ Frequency Shift at bad time
- Note: Requires frequency lock loop
- Difficulty: Medium
- Points: 61
- Solution: `flag{14142135}`

### Flag 6

- Modulation: AM USB w/ Frequency Shift at bad time
- Note: Requires frequency lock loop
- Difficulty: Medium
- Points: 67
- Solution: `flag{1732050807}`

### Flag 7

- Modulation: M17 Audio
- Difficulty: Very Hard due to frequency tracking
- Points: 97
- Solution: `flag{california}`

### Flag 8

- Modulation: M17 Metadata
- Difficulty: Very Hard due to frequency tracking
- Points: 61
- Solution: `flag{4601}`

### Flag 9

- Modulation: AM HD-Radio Broadcast
- Difficulty: Very Hard due to frequency tracking
- Points: 107
- Note: Digital audio different than analog audio.
- Solution: `flag{stillalive}`

### Flag 10

- Modulation: AM HD-Radio Metadata (station slogan)
- Difficulty: Very Hard due to frequency tracking
- Points: 43
- Solution: `flag{nrsc5_is_nice}`

### Flag 11

- Modulation: Morse
- Difficulty: Easy-Medium due to frequency shifting
- Points: 19
- Solution: `flag{superheterodyne}`
