"""Commandline client to run benchmarks"""
import click


@click.command()
def cli():
    pass


@click.group()
def benchmark():
    pass


if __name__ == '__main__':
    cli()
