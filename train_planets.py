import pathlib
from typing import Optional

import typer

from integrators.integrator_enum import IntegratorEnum

app = typer.Typer(add_completion=False)


@app.command()
def pointmass(run_name: Optional[str] = None, remake_dset: bool = False, exp: str = typer.Option(...)) -> None:
    from rnn_sophax.train_planets_runner import run_pointmass

    run_pointmass(run_name, remake_dset, exp)


@app.command()
def rigid(
    run_name: str,
    weights: pathlib.Path = typer.Option(None),
    exp: str = typer.Option(...),
    resume: Optional[pathlib.Path] = typer.Option(None),
    start_from: Optional[pathlib.Path] = typer.Option(None),
    integrator: IntegratorEnum = typer.Option(IntegratorEnum.VERLET)
) -> None:
    args = locals()

    from rnn_sophax.train_planets_runner import run_rigid
    from rnn_sophax.warmstart import WarmstartType

    if weights is not None and not weights.exists():
        typer.echo("Weights {} don't exist!".format(weights))
        raise typer.Exit(1)

    if resume is not None and not resume.exists():
        typer.echo("Resume {} doesn't exist!".format(resume))
        raise typer.Exit(1)

    if start_from is not None and not start_from.exists():
        typer.echo("Resume {} doesn't exist!".format(resume))
        raise typer.Exit(1)

    if sum([weights is not None, resume is not None, start_from is not None]) > 1:
        typer.echo("weights, resume and start_from are mutually exclusive!")
        raise typer.Exit(1)

    warmstart = None
    if weights is not None:
        warmstart = (weights, WarmstartType.Point)
    elif start_from is not None:
        warmstart = (start_from, WarmstartType.StartFrom)
    elif resume is not None:
        warmstart = (resume, WarmstartType.Resume)

    run_rigid(run_name, exp, warmstart, integrator, args)


if __name__ == "__main__":
    import ipdb

    with ipdb.launch_ipdb_on_exception():
        app()
