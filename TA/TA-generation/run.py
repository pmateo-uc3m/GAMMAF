import argparse
import logging
import os
import sys

from data_io import load_source_dataset
from pipeline import EnrichmentGenerator, load_config, select_entries, write_failures_report
from validation import validate_dataset


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Enrich InjecAgent test-case instances with benign tool "
        "responses and OpenAI tool descriptions using a vLLM-served model."
    )
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate input, select entries, and log the plan without calling the LLM "
        "or writing output",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="run a small test generation using test.num_entries",
    )
    parser.add_argument("--input", default=None, help="override input dataset path")
    parser.add_argument("--output", default=None, help="override output file path")
    parser.add_argument("--num-entries", type=int, default=None, help="override num_entries")
    parser.add_argument("--seed", type=int, default=None, help="override random_seed")
    return parser.parse_args(argv)


def setup_logging(cfg):
    level = getattr(logging, cfg["logging"].get("level", "INFO").upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    log_file = cfg["logging"].get("log_file")
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("TA-generation")


def resolve_output_path(cfg, override):
    if override:
        return override
    out_dir = cfg["output"]["path"]
    filename = cfg["output"]["filename"]
    return os.path.join(out_dir, filename)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    logger = setup_logging(cfg)

    num_entries = args.num_entries if args.num_entries is not None else cfg["num_entries"]
    random_seed = args.seed if args.seed is not None else cfg["random_seed"]

    input_cfg = dict(cfg["input"])
    if args.input:
        input_cfg["path"] = args.input

    dataset, source_file = load_source_dataset(input_cfg)
    logger.info("discovered %d source entries in %s", len(dataset), source_file)

    indices = select_entries(dataset, num_entries, random_seed)
    logger.info(
        "selected %d entries with seed=%d: source indices %s",
        len(indices), random_seed, indices,
    )

    if args.dry_run:
        logger.info("dry-run complete; no LLM calls and no output were produced")
        return 0

    if args.test:
        test_num = cfg["test"].get("num_entries", min(2, num_entries))
        if test_num > num_entries:
            logger.warning(
                "test.num_entries (%d) exceeds requested num_entries (%d); clamping",
                test_num, num_entries,
            )
            test_num = num_entries
        indices = indices[:test_num]
        output_path = os.path.join(
            cfg["output"]["path"], cfg["test"].get("output_filename", "enriched_dataset_test.json")
        )
        if args.output:
            output_path = args.output
        logger.info("TEST run: processing %d entries -> %s", len(indices), output_path)
    else:
        output_path = resolve_output_path(cfg, args.output)

    generator = EnrichmentGenerator(cfg, logger=logger)
    entries = generator.process(dataset, indices, source_file, output_path)

    if generator.failures:
        failures_path = os.path.join(
            os.path.dirname(output_path) or ".", "failures.json"
        )
        write_failures_report(generator.failures, failures_path)
        logger.warning(
            "%d generation(s) failed; see %s", len(generator.failures), failures_path
        )
        for failure in generator.failures:
            logger.error("failed source index %d: %s", failure["source_index"], failure["error"])

    logger.info(
        "done: %d/%d entries written to %s",
        len(entries), len(indices), output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
