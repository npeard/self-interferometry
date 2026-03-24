import marimo

__generated_with = '0.20.4'
app = marimo.App(width='medium')


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

    Protection diodes across the LD:
    - **Schottky** (anti-parallel): clamps reverse voltage across LD
    - **Zener** (parallel): clamps forward voltage across LD
    """)


@app.cell
def _(elm, schemdraw):
    PROT_COLOR = '#00000060'

    with schemdraw.Drawing() as d:
        d.config(unit=3)

        # Pin 2 at top (PD cathode / LD anode, floating +V)
        pin2 = d.add(elm.Dot(open=True).label('Pin 2 (+V)', loc='top'))

        # --- Right branch: LD + protection diodes ---
        d.add(elm.Line().at(pin2.center).right().length(2))
        top_node = d.add(elm.Dot())

        # LD (leftmost path, going down)
        d.add(elm.Diode().at(top_node.center).down().label('LD', loc='top'))
        bot_node = d.add(elm.Dot())

        # Schottky (anti-parallel to LD, offset right)
        d.add(elm.Line().at(top_node.center).right().length(1.5).color(PROT_COLOR))
        d.add(
            elm.Schottky()
            .down()
            .reverse()
            .color(PROT_COLOR)
            .label('Schottky', loc='right')
        )
        d.add(elm.Line().left().tox(bot_node.center[0]).color(PROT_COLOR))

        # Zener (parallel to LD, offset further right)
        d.add(elm.Line().at(top_node.center).right().length(3.0).color(PROT_COLOR))
        d.add(elm.Zener().down().color(PROT_COLOR).label('Zener', loc='right'))
        d.add(elm.Line().left().tox(bot_node.center[0]).color(PROT_COLOR))

        # --- Left branch: PD + R ---
        d.add(elm.Line().at(pin2.center).left().length(2))
        d.add(elm.Photodiode().down().reverse().label('PD', loc='top'))

        # Junction between PD and resistor (probe point)
        junc = d.add(elm.Dot())

        R_end = d.add(elm.Resistor().down().label('51 kΩ', loc='bottom'))

        # Probe from junction left to oscilloscope
        d.add(elm.Line().at(junc.center).left().length(2))
        scope = d.add(
            elm.Oscilloscope(signal='sine').label('Red Pitaya DAQ').anchor('in1')
        )

        # --- Common ground bus ---
        bus_y = R_end.end[1] - 0.5
        bus_left_x = scope.in2[0] - 1
        bus_right_x = bot_node.center[0]

        L_end = d.add(elm.Line().at(scope.in2).left().length(1))

        # Drop lines from each ground point to bus
        d.add(elm.Line().at(L_end.end).down().toy(bus_y))
        d.add(elm.Line().at(R_end.end).down().toy(bus_y))
        d.add(elm.Line().at(bot_node.center).down().toy(bus_y))

        # Horizontal ground bus
        d.add(elm.Line().at((bus_left_x, bus_y)).right().tox(bus_right_x).style(lw=3))

        # Single ground symbol at bus center
        bus_center_x = (bus_left_x + bus_right_x) / 2
        d.add(elm.Ground().at((bus_center_x, bus_y)))


@app.cell
def _():
    return


if __name__ == '__main__':
    app.run()
