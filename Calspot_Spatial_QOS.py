from meterapy4.audax.parser import LocalParser
#from meterapy4.plotter.metabolic import MetabolicPlotter

at_codes = {
    '@OP': 'na',
    '@PRJ': 'Spot Scan',
    '@CD': 'na',
    '@EXP': 'na'
}

# plotter = MetabolicPlotter
with LocalParser(supplied_at_codes=at_codes, fluidic_run=False, log_level='info') as meter:
    # plotter.set_logger(meter.tsv_log_file_path)
    
    meter.occonfig(0, 1300, 1500)
    meter.occonfig(1, 2100, 2300)
    meter.occonfig(2, 2900, 3100)
    meter.occonfig(3, 3700, 3900)

    meter.pv(15, 60, 60)
    meter.fmov(15, 0)
    meter.play(2, 100)
    meter.stripin()
    meter.play(1, 100)
    meter.pv(15, 0, 60)
    meter.delay(2000)
    meter.htr(1, 42, 120000)

    meter.sscparam(0, 20, 2000, 92000, 2, 520, 650, 1, 4)

    for j in range(4):
        meter.spsc(j,0,10,1)


        for i in range(20):
            #each spsc buffer is in a order ch 0 dry ch 0 wet... spsc(0,0) spsc(0,1). with 300 scans in each so spsc(0,0) = 0-300 scan options for SPD, spsc(0,1) = 301-600.. etc all the way up hence the 600 up to ch 3... each scan can have up to 300 individual scan 0-300 is whole buffer space of ch0 dry
            meter.spd(600*j+i)

        meter.delay(250)

    meter.spsc(0, 0, 10, 1)
    meter.spsc(1, 0, 10, 1)
    meter.spsc(2, 0, 10, 1)
    meter.spsc(3, 0, 10, 1)

    meter.qos(0, 0, 555, 565, 1, 0, 0, 0, 1, 15, 0)
    meter.qos(1, 0, 555, 565, 1, 0, 0, 0, 1, 15, 0)
    meter.qos(2, 0, 555, 565, 1, 0, 0, 0, 1, 15, 0)
    meter.qos(3, 0, 555, 565, 1, 0, 0, 0, 1, 15, 0)

    meter.pv(15, 60, 60)
