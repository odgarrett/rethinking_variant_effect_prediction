from pathlib import Path
import pandas as pd
import tempfile
import alignparse.targets
import alignparse.minimap2
import alignparse.consensus
from Bio import SeqIO

def expand_mutations(mut_string, wt_seq_list):
    """
    Reconstructs the full DNA sequence from a mutation string.

    Args:
        mut_string (str): e.g., "A52G T97C"
        wt_seq_list (list): The WT sequence as a list of characters (for mutability)
    """
    # Handle WT (Empty or NaN)
    if pd.isna(mut_string) or mut_string == "":
        return "".join(wt_seq_list)

    # Create a copy of the WT list to modify
    current_seq = wt_seq_list.copy()

    # Parse mutations
    # alignparse format is usually "Reference{Index}Variant" (e.g., A52G)
    try:
        for mut in mut_string.split():
            # Extract position (1-based index in the middle)
            # A52G -> starts at 1, ends at -1
            pos_str = mut[1:-1]
            new_base = mut[-1]
        
            # Convert to 0-based index
            idx = int(pos_str) - 1
        
            # Apply mutation
            if 0 <= idx < len(current_seq):
                current_seq[idx] = new_base
    except ValueError:
        # Fallback for weird parsing errors, return WT or flag it
        return "".join(wt_seq_list)
        
    return "".join(current_seq)

def align_and_parse(
        amplicon_file: str, 
        alignparse_config: str, 
        locus_name: str, 
        merged_reads_fastq: str, 
        alignment_path: str
        ) -> pd.DataFrame:
    '''
    Align sequencing reads to an amplicon reference and parse the alignment into a DataFrame.
    
    This function performs read-to-reference alignment using minimap2 and then parses the resulting
    alignment SAM file to extract and filter aligned sequences. The parsed alignment for a specific
    locus is returned as a pandas DataFrame.
    
    :param amplicon_file: Path to the FASTA file containing the reference amplicon sequences
    :param alignparse_config: Feature parsing specification configuration for alignparse
    :param locus_name: Name of the locus/target to extract from the parsed alignment results
    :param merged_reads_fastq: Path to the FASTQ file containing the merged sequencing reads to align
    :param alignment_path: Path where the SAM alignment file will be written
    :return: pandas.DataFrame containing the parsed alignment data for the specified locus
    '''
    
    # Load amplicon reference
    # This creates the main 'targets' object from your two files
    targets = alignparse.targets.Targets(
        seqsfile=amplicon_file, 
        feature_parse_specs=alignparse_config
    )

    # Set up the Aligner
    mapper = alignparse.minimap2.Mapper(alignparse.minimap2.OPTIONS_CODON_DMS)

    # Call targets.align() to run minimap2 and save the SAM file
    print(f"    Aligning reads from {merged_reads_fastq} to {amplicon_file}...")
    targets.align(
        queryfile=merged_reads_fastq,
        alignmentfile=alignment_path,
        mapper=mapper
    )

    # Parse alignment
    # Call targets.parse_alignment() to read the SAM file and apply filters
    print(f"    Parsing alignment from {alignment_path}...")
    readstats, aligned, filtered = targets.parse_alignment(
        alignment_path, 
        filtered_cs=True  # This tells what mutation reads had that were filtered out
    )

    # Update aligned DataFrame with mutation info columns
    print(f"    Adding mutation info columns for locus {locus_name}...")
    aligned_df = alignparse.consensus.add_mut_info_cols(
        aligned[locus_name],
        mutation_col="gene_mutations",
        n_sub_col="n_subs",
        n_indel_col="n_indels",
        overwrite_cols=True,
    )

    return pd.DataFrame(aligned_df)


def construct_master_alignment_df(
        amplicon_file: str, 
        alignparse_config: str, 
        locus_name: str, 
        merged_reads_dir: Path, 
        ) -> pd.DataFrame:
    '''
    Construct a master DataFrame by aligning and parsing all FASTQ files from a directory.
    
    This function processes all merged reads FASTQ files in a specified directory, aligns them
    to the amplicon reference, and consolidates the results into a single master DataFrame.
    Reads with indels or ambiguous bases (N) are filtered out, and sample metadata is added
    to each record.
    
    :param amplicon_file: Path to the FASTA file containing the reference amplicon sequences
    :param alignparse_config: Feature parsing specification configuration for alignparse
    :param locus_name: Name of the locus/target to extract from alignment results
    :param merged_reads_dir: Path to directory containing FASTQ files to process. Sample names
                            are extracted from the first part of each filename (before the first underscore)
    :return: pandas.DataFrame containing consolidated alignment results across all samples, with columns:
            - library: Library identifier (all reads belong to one physical library)
            - sample: Sample name extracted from the FASTQ filename
            - query_sequence: The aligned sequence
            - gene_mutations: Mutations detected in the gene
    '''
    
    all_sample_dfs = []

    for fastq_file in merged_reads_dir.glob("*.fastq"):
        sample_name = fastq_file.name.split("_")[0]
        print(f"\nProcessing {sample_name}...")

        # Use temp file for SAM output to save disk space
        with tempfile.NamedTemporaryFile(suffix=".sam") as temp_sam:
            aligned_df = align_and_parse(
                amplicon_file=amplicon_file,
                alignparse_config=alignparse_config,
                locus_name=locus_name,
                merged_reads_fastq=str(fastq_file),
                alignment_path=temp_sam.name
            )

        # Remove indels and Ns
        aligned_df = aligned_df.query("n_indels == 0 and not gene_mutations.str.contains('N')").copy()

        # Load WT sequence as list for mutation expansion
        wt_record = SeqIO.read(amplicon_file, "genbank")
        wt_dna_list = list(str(wt_record.seq))

        # Expand mutations to full sequences
        print(f"    Expanding mutations to full sequences for {sample_name}...")
        aligned_df['sequence'] = aligned_df['gene_mutations'].apply(
            lambda x: expand_mutations(x, wt_dna_list)
        )

        # Add metadata
        aligned_df['library'] = 'library' # everything belongs to one physical library
        aligned_df['sample'] = sample_name

        # Append only necessary columns to master list
        cols_to_keep = ['library', 'sample', 'sequence', 'gene_mutations']
        all_sample_dfs.append(aligned_df[cols_to_keep])

    # Concatenate all sample DataFrames into a master DataFrame
    print("Combining all sample DataFrames into master DataFrame...")
    master_df = pd.concat(all_sample_dfs, ignore_index=True)
    
    print(f'Total aligned reads across all samples: {len(master_df)}')

    return master_df