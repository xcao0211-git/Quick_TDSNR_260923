Single channel analytic reference: matched 6.0206 dB pad + 500 ps line
* Run HSPICE from this directory. V(src) is OPEN-CIRCUIT Thevenin voltage.
* PWL exactly matches Quick_TDSNR samples: UI=100ps rise=20ps dt=1ps.
.option post=2 accurate reltol=1e-6 vntol=1e-9 delmax=0.1p
Vdrive src 0 PWL(0 0 20p 1 100p 1 120p 0 4096p 0) AC 1
Rsource src tx 50
* Physical channel begins at tx; these resistors are INTERNAL to the channel.
Rpad1 tx mid 16.66666666666667
Rpadsh mid 0 66.66666666666667
Rpad2 mid linein 16.66666666666667
Tchannel linein 0 rx 0 Z0=50 TD=500p
* Physical channel ends at rx.
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
