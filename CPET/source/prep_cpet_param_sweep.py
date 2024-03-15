import os
import shutil
import argparse

def create_sweep_files(sample_list, 
                       stepsize_list, 
                       pdb, 
                       output, 
                       center_xyz, 
                       axis_1, 
                       axis_2,
                       box_size=2.0,
                       bins=20,
                       replicate=3):
    #Create the sweep files
    for i in sample_list:
        for j in stepsize_list:
            for k in range(replicate):
                j=j.split('.')[1]
                name=f'{str(i)}_{str(j)}_{str(k)}'
                pdb_name=output+pdb.split('/')[-1].split('.')[0]+"_"+name+".pdb"
                options_name=output+"options"+"_"+name+".txt"
                
                #Save pdb file with pdb_name:
                shutil.copy(pdb, pdb_name)

                #Modify options file:
                with open(options_name, 'w') as f:
                        f.write(
                            f"align {center_xyz[0]}:{center_xyz[1]}:{center_xyz[2]} {axis_1[0]}:{axis_1[1]}:{axis_1[2]} {axis_2[0]}:{axis_2[1]}:{axis_2[2]}\n"
                        )
                        f.write(f"%topology \n")
                        f.write(
                            "    volume box {} {} {} \n".format(
                                box_size, box_size, box_size
                            )
                        )
                        f.write("    stepSize {} \n".format(j))
                        f.write("    samples {} \n".format(i))
                        f.write(
                            "    sampleOutput output_{}_{}_{} \n".format(
                                str(i), str(j)[2:], str(k)
                            )
                        )
                        f.write("    bins {} \n".format(bins))
                        f.write(f"end \n")
                        f.close()



def main():
    parser = argparse.ArgumentParser(description="Prepare sweep")
    parser.add_argument("--sample_list", type=list, help="List of samples", default=[10,100,1000,10000,100000])
    parser.add_argument("--stepsize_list", type=list, help="List of stepsizes", default=[0.5,0.1,0.05,0.01,0.001])
    parser.add_argument("--pdb", type=str, help="Path to the pdb file", default='./test.pdb')
    parser.add_argument("--options", type=str, help="Path to the options file", default='./options_test.txt')
    parser.add_argument("--replicate", type=int, help="Number of replicates", default=3)
    parser.add_argument("--output", type=str, help="Output directory", default='./cpet_convergence/')
    args = parser.parse_args()
    center_xyz = [1,2,3]
    axis_1 = [1,2,3]
    axis_2 = [1,2,3]
    create_sweep_files(args.sample_list, 
                       args.stepsize_list, 
                       args.pdb, 
                       args.output, 
                       center_xyz, 
                       axis_1, 
                       axis_2,
                       box_size=2.0,
                       bins=20,
                       replicate=3)
    
if __name__=="__main__":
    main()

