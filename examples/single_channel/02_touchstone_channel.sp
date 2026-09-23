Single channel Touchstone comparison: same source and terminations
* Keep this netlist and the original .s2p in the same working directory.
.option post=2 accurate reltol=1e-6 vntol=1e-9 delmax=0.1p
Vdrive src 0 PWL(0 0 20p 1 100p 1 120p 0 4096p 0) AC 1
Rsource src tx 50
* Two signal nodes followed by the COMMON reference node, not four terminals.
Schannel tx rx 0 mname=channel_model
+ INTERPOLATION=LINEAR INTDATTYP=MA
.model channel_model S N=2
+ TSTONEFILE='single_channel_50ohm_500ps_6dB.s2p'
+ FBASE=100meg FMAX=500g
Rload rx 0 50
.tran 1p 4096p
.print tran v(src) v(tx) v(rx)
.probe tran v(src) v(tx) v(rx)
.measure tran rx_peak MAX v(rx) FROM=450p TO=650p
.measure tran rx_mid FIND v(rx) AT=560p
.measure tran flight TRIG v(src) VAL=0.5 RISE=1
+ TARG v(rx) VAL=0.125 RISE=1
.ac lin 5001 1meg 500g
.print ac vdb(rx) vp(rx)
.end
