import pandas as pd
import tempfile
from Bio import SeqIO
import dms_variants.codonvarianttable

def construct_variant_table(
        aligned_reads_df: pd.DataFrame,
        wt_dna_file: str,
        ) -> dms_variants.codonvarianttable.CodonVariantTable:
    print("Constructing variant table...")

    # Rename columns to match dms_variants expectations, using DNA sequence as the "barcode"
    formatted_df = aligned_reads_df.rename(columns={
        'sequence': 'barcode',
        'gene_mutations': 'substitutions'
    })

    # Add dummy variant call support column
    formatted_df['variant_call_support'] = 1

    # Create barcode definition file expected by dms_variants
    definition_df = formatted_df[['library', 'barcode', 'substitutions', 'variant_call_support']].drop_duplicates(subset='barcode')

    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as temp_def:
        definition_df.to_csv(temp_def.name, index=False)
        definition_path = temp_def.name

    # Load WT sequence
    wt_record = SeqIO.read(wt_dna_file, "fasta")
    wt_seq = str(wt_record.seq)

    # Initialize CodonVariantTable
    variant_table = dms_variants.codonvarianttable.CodonVariantTable(
        barcode_variant_file=definition_path,
        geneseq=wt_seq,
        substitutions_col='substitutions',
        allowgaps=False
    )

    # Count variants and add counts to variant table
    counts = formatted_df.groupby(['library', 'sample', 'barcode']).size().reset_index(name='count')

    variant_table.add_sample_counts_df(counts)
    return variant_table
