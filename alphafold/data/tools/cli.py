# -------------------- Functional interface for Click CLI ----------------------
import os
import json
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

DEFAULT_CACHE_DIR = '.cache'


@click.group()
def cli():
    pass


def parse_fasta(input_fasta_path):
  """Given fasta file at `input_fasta_path`, calls `parsers.parse_fasta`
  returning tuple of input sequence, input description, and sequence length.
  """
  with open(input_fasta_path) as f:
      input_fasta_str = f.read()
  input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
  return (input_seqs[0], input_descs[0], len(input_sequence))


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
    return hashlib.md5(s).hexdigest()


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
    obj = (args, OrderedDict(kwargs.items()))
    serialized = json.dumps(obj).encode('utf-8')
    return md5_hash(serialized)
    

def cache_to_pckl(cache_dir=DEFAULT_CACHE_DIR, exclude_kw=None, use_pckl=True):
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
    
    # whether to use pickle or plain text cache
    cache_ext = 'pckl' if use_pckl is True else 'out'
    SKIP_PCKL_CACHE = os.environ.get('SKIP_PCKL_CACHE', 0)

    def decorator(fn):
        def wrapped(*args, **kwargs):
            kw = {k: kwargs[k] for k in kwargs if k not in exclude_kw}
            key = cache_key(args, kw)
            cache_fp = os.path.join(cache_dir, fn.__name__, f"{key}.{cache_ext}")

            logging.info(f"using cache_fp={cache_fp}")
            logging.info(f"SKIP_PCKL_CACHE={SKIP_PCKL_CACHE}")

            if os.path.exists(cache_fp) and not SKIP_PCKL_CACHE:
                logging.info(f"using cache at {cache_fp} instead of running {fn.__name__}")
                if use_pckl:
                    with open(cache_fp, 'rb') as f:
                        return pickle.load(f)
                else:
                    with open(cache_fp, 'r') as f:
                        return f.read()

            result = fn(*args, **kwargs)

            # write to cache file
            os.makedirs(os.path.dirname(cache_fp), exist_ok=True)
            logging.info(f"saving results to {cache_fp}")
            if use_pckl:
                with open(cache_fp, 'wb') as f:
                    pickle.dump(result, f)
            else:
                with open(cache_fp, 'w') as f:
                    f.write(result)

            return result

        return wrapped

    return decorator

# ------------------------------------------------------------------------------

@cli.command(name='jackhmmer_uniref90')
@click.option('--input-fasta-path', required=True, type=click.Path())
@click.option('--jackhmmer-binary-path', required=True, type=click.Path())
@click.option('--uniref90-database-path', required=True, type=click.Path())
@click.option('--output-dir', required=True, type=click.Path(file_okay=False))
def jackhmmer_uniref90_cli(*args, **kwargs):
    return jackhmmer_uniref90(*args, **kwargs)


@cache_to_pckl(exclude_kw='output_dir')
def jackhmmer_uniref90(input_fasta_path: str, jackhmmer_binary_path: str,
                       uniref90_database_path: str, output_dir: str):
  jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
      binary_path=jackhmmer_binary_path,
      database_path=uniref90_database_path)
  result = jackhmmer_uniref90_runner.query(input_fasta_path)[0]['sto']
  return write_output(result, 'uniref90_hits.sto', output_dir=output_dir)


@cli.command(name='jackhmmer_mgnify')
@click.option('--input-fasta-path', required=True, type=click.Path())
@click.option('--jackhmmer-binary-path', required=True, type=click.Path())
@click.option('--mgnify-database-path', required=True, type=click.Path())
@click.option('--mgnify-max-hits', required=True, type=click.INT)
@click.option('--output-dir', required=True, type=click.Path(file_okay=False))
def jackhmmer_mgnify_cli(*args, **kwargs):
    return jackhmmer_mgnify(*args, **kwargs)


@cache_to_pckl(exclude_kw='output_dir')
def jackhmmer_mgnify(input_fasta_path: str, jackhmmer_binary_path: str,
                     mgnify_database_path: str, mgnify_max_hits, 
                     output_dir: str):
  jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
      binary_path=jackhmmer_binary_path,
      database_path=mgnify_database_path)
  result = jackhmmer_mgnify_runner.query(input_fasta_path)[0]['sto']
  write_output(result, 'mgnify_hits.sto', output_dir=output_dir)

  mgnify_msa, mgnify_deletion_matrix, _ = parsers.parse_stockholm(result)
  mgnify_msa = mgnify_msa[:mgnify_max_hits]
  mgnify_deletion_matrix = mgnify_deletion_matrix[:mgnify_max_hits]

  return (mgnify_msa, mgnify_deletion_matrix)


@cache_to_pckl(exclude_kw='output_dir')
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
  return write_output(result, 'pdb70_hits.hhr', output_dir=output_dir)


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


@cache_to_pckl()
def template_featurize(input_fasta_path, hhsearch_hits_path, mmcif_dir: str,
                       max_template_date, max_hits, kalign_binary_path,
                       release_dates_path, obsolete_pdbs_path,
                       strict_error_check):
  with open(hhsearch_hits_path, 'r') as f:
    hhsearch_hits = parsers.parse_hhr(f.read())
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