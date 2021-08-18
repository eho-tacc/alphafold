"""Modular version of alphafold.data.pipeline"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
from absl import logging
from alphafold.data import parsers
from alphafold.data import templates
from alphafold.data.tools.cli import *
from alphafold.data.pipeline import make_sequence_features, make_msa_features
import numpy as np

# Internal import (7716).

FeatureDict = Mapping[str, np.ndarray]


@dataclass
class ModularDataPipeline:
  """Modular version of alphafold.data.pipeline.DataPipeline"""
  jackhmmer_binary_path: str
  hhblits_binary_path: str
  hhsearch_binary_path: str
  uniref90_database_path: str
  mgnify_database_path: str
  pdb70_database_path: str
  use_small_bfd: bool

  # for construction of TemplateHitFeaturizer, replacing
  # template_featurizer: templates.TemplateHitFeaturizer
  mmcif_dir: str
  max_template_date: str
  max_hits: int
  kalign_binary_path: str
  release_dates_path: str = None
  obsolete_pdbs_path: str = None
  strict_error_check: bool = False

  mgnify_max_hits: int = 501
  uniref_max_hits: int = 10000
  bfd_database_path: str = None
  uniclust30_database_path: str = None
  small_bfd_database_path: str = None

  def init_runners(self,
               jackhmmer_binary_path: str,
               hhblits_binary_path: str,
               hhsearch_binary_path: str,
               uniref90_database_path: str,
               mgnify_database_path: str,
               bfd_database_path: Optional[str],
               uniclust30_database_path: Optional[str],
               small_bfd_database_path: Optional[str],
               pdb70_database_path: str,
               template_featurizer: templates.TemplateHitFeaturizer,
               use_small_bfd: bool,
               mgnify_max_hits: int = 501,
               uniref_max_hits: int = 10000):
    """Deprecated"""
    pass
  
  def out_path(self, fname):
      return os.path.join(self.msa_output_dir, fname)
    
  def write(self, data, fname) -> str:
    fp = self.out_path(fname)
    with open(fp, 'w') as f:
      f.write(data)
    return fp

  def load_input_fasta(self, input_fasta_path):
    with open(input_fasta_path) as f:
      input_fasta_str = f.read()
    input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)
    if len(input_seqs) != 1:
      raise ValueError(
          f'More than one input sequence found in {input_fasta_path}.')
    input_sequence = input_seqs[0]
    input_description = input_descs[0]
    return (input_sequence, input_description)

  def jackhmmer_uniref90(self, input_fasta_path: str):
    jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
        binary_path=self.jackhmmer_binary_path,
        database_path=self.uniref90_database_path)
    result = jackhmmer_uniref90_runner.query(input_fasta_path)[0]['sto']
    return self.write(result, 'uniref90_hits.sto')

  def jackhmmer_mgnify(self, input_fasta_path: str):
    return jackhmmer_mgnify(
      input_fasta_path=input_fasta_path, 
      jackhmmer_binary_path=self.jackhmmer_binary_path, 
      mgnify_database_path=self.mgnify_database_path, 
      mgnify_max_hits=self.mgnify_max_hits,
      output_dir=self.msa_output_dir
    )
  
  def hhsearch_pdb70(self, jackhmmer_uniref90_result_path):
    return hhsearch_pdb70(
      jackhmmer_uniref90_result_path=jackhmmer_uniref90_result_path, 
      hhsearch_binary_path=self.hhsearch_binary_path,
      pdb70_database_path=self.pdb70_database_path, 
      uniref_max_hits=self.uniref_max_hits,
      output_dir=self.msa_output_dir
    )
  
  def jackhmmer_small_bfd(self, input_fasta_path):
    return jackhmmer_small_bfd(
      input_fasta_path=input_fasta_path, 
      jackhmmer_binary_path=self.jackhmmer_binary_path, 
      small_bfd_database_path=self.small_bfd_database_path,
      output_dir=self.msa_output_dir
    )

  def hhblits(self, input_fasta_path):
    return hhblits(
      input_fasta_path=input_fasta_path, 
      hhblits_binary_path=self.hhblits_binary_path,
      bfd_database_path=self.bfd_database_path,
      uniclust30_database_path=self.uniclust30_database_path,
      output_dir=self.msa_output_dir
    )

  def template_featurize(self, input_fasta_path, hhsearch_hits_path):
    return hhblits(
      input_fasta_path=input_fasta_path, 
      hhsearch_hits_path=hhsearch_hits_path,
      mmcif_dir=self.mmcif_dir,
      max_template_date=self.max_template_date, 
      max_hits=self.max_hits, 
      kalign_binary_path=self.kalign_binary_path,
      release_dates_path=self.release_dates_path, 
      obsolete_pdbs_path=self.obsolete_pdbs_path,
      strict_error_check=self.strict_error_check,
      output_dir=self.msa_output_dir
    )

  def process(self, input_fasta_path: str, msa_output_dir: str) -> FeatureDict:
    """Runs alignment tools on the input sequence and creates features."""
    self.msa_output_dir = msa_output_dir
    input_sequence, input_description = self.load_input_fasta(input_fasta_path)
    sequence_features = make_sequence_features(
        sequence=input_sequence,
        description=input_description,
        num_res=len(input_sequence))

    jackhmmer_uniref90_result_path = self.jackhmmer_uniref90(input_fasta_path)
    mgnify_msa, mgnify_deletion_matrix = self.jackhmmer_mgnify(input_fasta_path)
    hhsearch_hits_path = self.hhsearch_pdb70(jackhmmer_uniref90_result_path)

    with open(jackhmmer_uniref90_result_path, 'r') as f:
      uniref90_msa, uniref90_deletion_matrix, _ = parsers.parse_stockholm(f.read())

    if self.use_small_bfd:
      jackhmmer_small_bfd_result = self.jackhmmer_small_bfd(input_fasta_path)
      bfd_msa, bfd_deletion_matrix, _ = parsers.parse_stockholm(
          jackhmmer_small_bfd_result)
    else:
      hhblits_bfd_uniclust_result = self.hhblits(input_fasta_path)
      bfd_msa, bfd_deletion_matrix = parsers.parse_a3m(
          hhblits_bfd_uniclust_result)

    templates_result = self.template_featurize(input_sequence, hhsearch_hits_path)

    msa_features = make_msa_features(
        msas=(uniref90_msa, bfd_msa, mgnify_msa),
        deletion_matrices=(uniref90_deletion_matrix,
                           bfd_deletion_matrix,
                           mgnify_deletion_matrix))

    logging.info('Uniref90 MSA size: %d sequences.', len(uniref90_msa))
    logging.info('BFD MSA size: %d sequences.', len(bfd_msa))
    logging.info('MGnify MSA size: %d sequences.', len(mgnify_msa))
    logging.info('Final (deduplicated) MSA size: %d sequences.',
                 msa_features['num_alignments'][0])
    logging.info('Total number of templates (NB: this can include bad '
                 'templates and is later filtered to top 4): %d.',
                 templates_result.features['template_domain_names'].shape[0])

    return {**sequence_features, **msa_features, **templates_result.features}
