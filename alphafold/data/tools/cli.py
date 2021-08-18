# -------------------- Functional interface for Click CLI ----------------------
import os
import click
from collections import OrderedDict
import pickle
from functools import lru_cache
import hashlib
from typing import Mapping, Optional, Sequence
from alphafold.data.tools import hhblits
from alphafold.data.tools import hhsearch
from alphafold.data.tools import jackhmmer
from alphafold.data import parsers
from alphafold.data import templates

import logging
logging.basicConfig(level=logging.DEBUG)


@click.group()
def cli():
    pass


def write_output(data, fname, output_dir) -> str:
  fp = os.path.join(output_dir, fname)
  with open(fp, 'w') as f:
    f.write(data)
  return fp


@lru_cache(maxsize=32)
def hash_fp(fp):
    with open(fp, 'rb') as f:
        return md5_hash(f.read())


def md5_hash(s: str) -> str:
    hasher = hashlib.md5()
    hasher.update(s)
    return hasher.hexdigest()


def same_hash(s1: str, s2: str) -> bool:
    return md5_hash(s1) == md5_has(s2)


def cache_key_with_hashed_paths(args, kwargs):
    """Hashable key of function name, args, and kwargs. For kwarg names ending
    in '_path', attempt to hash the file and set that as the key.
    """
    kw = dict()
    for k in kwargs:
        assert isinstance(k, str)
        if k.endswith('_path'):
            kw[k] = hash_fp(kwargs[k])
        else:
            kw[k] = kwargs[k]
    return (args, frozenset(kw.items()))


def cache_key(args, kwargs):
    return (args, frozenset(kwargs.items()))


def cache_to_pckl(cache_dir='.cache', exclude_kw=None):
    """Caches function results to pickle file. Returns a decorator factory.
    Pickled function results are cached to a path `cache_dir/func.__name__/hash`
    where hash is hashed args and kwargs (except kwargs listed in `exclude_kw`).
    If the cache file exists, return the unpickled result. If it does not exist,
    or if environment variable `SKIP_PCKL_CACHE=1`, run the function and write
    its result to the cache.
    """
    if exclude_kw is None:
        exclude_kw = list()
    elif isinstance(exclude_kw, str):
        exclude_kw = [exclude_kw]

    def decorator(fn):
        def wrapped(*args, **kwargs):
            key = hash(cache_key(set(args), OrderedDict({k: kwargs[k] for k in kwargs if k not in exclude_kw})))
            cache_fp = os.path.join(cache_dir, fn.__name__, f"{key}.pckl")
            SKIP_PCKL_CACHE = os.environ.get('SKIP_PCKL_CACHE', 0)
            logging.debug(f"using cache_fp={cache_fp}")
            logging.debug(f"SKIP_PCKL_CACHE={SKIP_PCKL_CACHE}")
            if os.path.exists(cache_fp) and not SKIP_PCKL_CACHE:
                logging.info(f"using cache at {cache_fp} instead of running {fn.__name__}")
                with open(cache_fp, 'rb') as f:
                    return pickle.load(f)

            result = fn(*args, **kwargs)

            # write to cache file
            os.makedirs(os.path.dirname(cache_fp), exist_ok=True)
            with open(cache_fp, 'wb') as f:
                logging.info(f"saving pickled results to {cache_fp}")
                pickle.dump(result, f)

            return result

        return wrapped

    return decorator

# ------------------------------------------------------------------------------

@cache_to_pckl(exclude_kw='output_dir')
def jackhmmer_uniref90(input_fasta_path: str, jackhmmer_binary_path: str,
                       uniref90_database_path: str, output_dir: str):
  jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
      binary_path=jackhmmer_binary_path,
      database_path=uniref90_database_path)
  result = jackhmmer_uniref90_runner.query(input_fasta_path)[0]['sto']
  write_output(result, 'uniref90_hits.sto', output_dir=output_dir)
  return result


@cache_to_pckl(exclude_kw='output_dir')
def jackhmmer_mgnify(input_fasta_path: str, jackhmmer_binary_path: str,
                     mgnify_database_path: str, output_dir: str):
  jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
      binary_path=jackhmmer_binary_path,
      database_path=mgnify_database_path)
  result = jackhmmer_mgnify_runner.query(input_fasta_path)[0]['sto']
  write_output(result, 'mgnify_hits.sto', output_dir=output_dir)
  return result


def hhsearch_pdb70(jackhmmer_uniref90_result_path, hhsearch_binary_path: str,
                   pdb70_database_path: str, output_dir: str,
                   uniref_max_hits):
  with open(jackhmmer_uniref90_result_path, 'r') as f:
    jackhmmer_uniref90_result = f.read()
  hhsearch_pdb70_runner = hhsearch.HHSearch(
      binary_path=hhsearch_binary_path,
      databases=[pdb70_database_path])
  uniref90_msa_as_a3m = parsers.convert_stockholm_to_a3m(
      jackhmmer_uniref90_result, max_sequences=uniref_max_hits)
  result = hhsearch_pdb70_runner.query(uniref90_msa_as_a3m)
  write_output(result, 'pdb70_hits.hhr', output_dir=output_dir)
  return parsers.parse_hhr(result)


@cache_to_pckl(exclude_kw='output_dir')
def jackhmmer_small_bfd(input_fasta_path: str, jackhmmer_binary_path: str,
                        small_bfd_database_path: str, output_dir: str):
  jackhmmer_small_bfd_runner = jackhmmer.Jackhmmer(
      binary_path=jackhmmer_binary_path,
      database_path=small_bfd_database_path)
  result = jackhmmer_small_bfd_runner.query(input_fasta_path)[0]['sto']
  write_output(result, 'small_bfd_hits.a3m', output_dir=output_dir)
  return result


@cache_to_pckl(exclude_kw='output_dir')
def hhblits(input_fasta_path: str, hhblits_binary_path: str,
            bfd_database_path: str, uniclust30_database_path: str,
            output_dir: str):
  hhblits_bfd_uniclust_runner = hhblits.HHBlits(
      binary_path=hhblits_binary_path,
      databases=[bfd_database_path, uniclust30_database_path])
  result = hhblits_bfd_uniclust_runner.query(input_fasta_path)['a3m']
  write_output(result, 'bfd_uniclust_hits.a3m', output_dir=output_dir)
  return result


def template_featurize(input_fasta_path, hhsearch_hits, mmcif_dir: str,
                       max_template_date, max_hits, kalign_binary_path,
                       release_dates_path, obsolete_pdbs_path,
                       strict_error_check):
  template_featurizer = templates.TemplateHitFeaturizer(
    mmcif_dir=mmcif_dir,
    max_template_date=max_template_date,
    max_hits=max_hits,
    kalign_binary_path=kalign_binary_path,
    release_dates_path=release_dates_path,
    obsolete_pdbs_path=obsolete_pdbs_path,
    strict_error_check=strict_error_check,
  )
  return template_featurizer.get_templates(
      query_sequence=input_sequence,
      query_pdb_code=None,
      query_release_date=None,
      hits=hhsearch_hits)


if __name__ == '__main__':
    cli()