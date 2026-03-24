import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import schemdraw
    import schemdraw.elements as elm

    return elm, mo, schemdraw


@app.cell
def _(mo):
    mo.md(r"""
    # Circuit Diagrams

    ## Type A Laser Diode / Photodiode Package

    A Type A package contains a laser diode (LD) and a monitor
    photodiode (PD) with opposite orientations through a shared pin:

    - **Pin 1**: PD anode → through 51 kΩ sense resistor → ground
    - **Pin 2**: PD cathode / LD anode (floating, positively biased)
    - **Pin 3**: LD cathode → ground

    Pin 2 is positively biased, forward-biasing the LD and
    reverse-biasing the PD. A 51 kΩ resistor converts PD photocurrent
    to a voltage measured by a ground-referenced oscilloscope probe.
    """)
    return


@app.cell
def _(elm, schemdraw):
    with schemdraw.Drawing() as d:
        d.config(unit=3)

        # Pin 2 (PD cathode / LD anode, floating +V)
        pin2 = d.add(elm.Dot(open=True).label('Pin 2\n(+V)', loc='left'))

        # Upper branch: LD (anode=Pin2 → cathode=Pin3 → GND)
        d.add(elm.Line().at(pin2.center).up().length(1.5))
        d.add(elm.Diode().right().label('LD', loc='top'))
        d.add(elm.Line().length(0.5))
        d.add(elm.Ground())

        # Lower branch: PD (reversed, cathode=Pin2 → anode) + R → GND
        d.add(elm.Line().at(pin2.center).down().length(1.5))
        d.add(elm.Photodiode().right().reverse().label('PD', loc='top'))

        # Junction between PD and resistor (probe point)
        junc = d.add(elm.Dot())

        d.add(elm.Resistor().right().label('51 kΩ', loc='top'))
        d.add(elm.Ground())

        # Probe from junction down to oscilloscope
        d.add(elm.Line().at(junc.center).down().length(1.5))
        scope = d.add(elm.Oscilloscope(signal='sine').anchor('in1'))
        d.add(elm.Line().at(scope.in2).down().length(0.3))
        d.add(elm.Ground())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
