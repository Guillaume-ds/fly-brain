from __future__ import annotations

import argparse
import logging

from . import analyze, data, simulate


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(prog="fly-brain", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download", help="pre-download connectome files")
    p_download.add_argument(
        "--full", action="store_true",
        help="also download the ~500MB connectivity weights file",
    )

    p_analyze = sub.add_parser("analyze", help="print/plot connectome-wide stats")
    p_analyze.add_argument(
        "--connectivity", action="store_true",
        help="also compute degree stats (downloads the ~500MB weights file)",
    )

    p_sim = sub.add_parser("simulate", help="run a toy signal-propagation demo on a real circuit")
    p_sim.add_argument("--seed-type", default="DNp01",
                        help="neuron 'type' to center the circuit on (default: DNp01, the Giant Fiber escape neuron)")
    p_sim.add_argument("--hops", type=int, default=2, help="how many synaptic hops downstream to expand")
    p_sim.add_argument("--max-neurons", type=int, default=60, help="cap on circuit size")
    p_sim.add_argument("--steps", type=int, default=40, help="simulation timesteps")
    p_sim.add_argument("--stim-steps", type=int, default=4, help="timesteps to inject stimulus for")
    p_sim.add_argument("--no-animate", action="store_true", help="skip the live terminal animation")

    args = parser.parse_args(argv)

    if args.command == "download":
        data.fetch("annotations")
        data.fetch("neurotransmitters")
        if args.full:
            data.fetch("weights")
    elif args.command == "analyze":
        analyze.run(with_connectivity=args.connectivity)
    elif args.command == "simulate":
        simulate.run(
            seed_type=args.seed_type,
            hops=args.hops,
            max_neurons=args.max_neurons,
            steps=args.steps,
            stim_steps=args.stim_steps,
            animate=not args.no_animate,
        )


if __name__ == "__main__":
    main()
