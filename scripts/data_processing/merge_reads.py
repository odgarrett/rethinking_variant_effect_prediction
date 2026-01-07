from pathlib import Path
import subprocess

def merge_reads(
        raw_reads_dir: Path, 
        merged_reads_dir: Path,
        ):
    '''
    Merge all reads in raw_reads_dir using vsearch. Assumes you've named 
    them with {sample_name}_{R1 or R2}.fastq.gz. 

    Creates {sample_name}_merged.fastq in merged_reads_dir.
    
    Key vsearch args:
        --fastq_minovlen = 20
        --fastq_maxee = 1.0

    :param raw_reads_dir: Path
    :param merged_reads_dir: Path
    '''
    merged_reads_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate through each R1 file
    for r1_path in raw_reads_dir.glob("*_R1.fastq.gz"):
        
        # Generate R2 filename
        r2_name = r1_path.name.replace("_R1", "_R2")
        r2_path = raw_reads_dir / r2_name

        # Generate output filename
        sample_name = r1_path.name.split('_')[0]
        output_filename = f'{sample_name}_merged.fastq'
        output_path = merged_reads_dir / output_filename

        # Build the command
        cmd = [
            "vsearch",
            "--fastq_mergepairs", str(r1_path),
            "--reverse", str(r2_path),
            "--fastq_minovlen", "20",
            "--fastq_maxee", "1.0",
            "--fastqout", str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
