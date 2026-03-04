import os

#input file for chg2cube.pl
with open('input.txt', '+w') as f:
    f.write('3\n')          #SELECT INPUT FILE FORMAT 1=CHGCAR or 2=LOCPOT 3=PARCHG:
    f.write('14 6 8 7 1\n')     #ATOMIC NUMBERS of atoms in POSCAR with the same order
    f.close()


for i in range(193,243):
    print("WORKING ON =====> PARCHG.{:04d}.ALLK_uniform".format(i))
    os.system('/gpfs/scratch/uam67/650249/Util/vtstscripts-1033/chg2cube.pl PARCHG.{:04d}_uniform < input.txt'.format(i)) 
