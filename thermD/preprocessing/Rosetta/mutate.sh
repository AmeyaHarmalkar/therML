#!/bin/bash

aa_list=(ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR)

#maxprocesses=10
set -e   # Exit script if an error occured

for res in {1..20}; do
        echo $res;

        for amino in ${aa_list[@]}; do
                echo $res;
                echo $amino;

                        $WORK/Rosetta/main/source/bin/rosetta_scripts.linuxclangrelease -in:file:s complex.pdb -parser:protocol mutate_relax_analyze.xml -parser:script_vars position=$res\H -parser:script_vars res_aa=$amino -in:file:native native.pdb -out:file:scorefile vH_score.sc -out:path:pdb out_pdbs -out:pdb_gz -out:suffix _$res\H\_$amino -nstruct 1 -overwrite | tee outerr/process_cH.log
                done
        done
done

echo "Finished mutational scan!"
