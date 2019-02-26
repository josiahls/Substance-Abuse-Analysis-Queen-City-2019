#! /bin/bash
# exec 1>$PBS_O_WORKDIR/out 2>$PBS_O_WORKDIR/err
#
# ===== PBS OPTIONS =====
#
#PBS -N "classifier_jlaivins"
#PBS -q copperhead
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=1:gpus=1:gtx1080ti,mem=16GB
#PBS -V
#
# ===== END PBS OPTIONS =====
#
# ==== Main ======
cd $PBS_O_WORKDIR
mkdir log
{
#module load cuda/8.0 cudnn/6.0-cuda8 pytorch/0.4.0-anaconda3-cuda8
module load pytorch/0.4.0-anaconda3-cuda9.2-sm6.1

python ./Driver.py
} > log/output_"$PBS_JOBNAME"_$PBS_JOBID 2>log/errorLog_"$PBS_JOBNAME"_$PBS_JOBID

## Running: Server
# These instructions are made with the UNCC server in mind. It also is intended that you use Linux, or Cmder
# 1. Log into the server via: `ssh [username]@hpc.uncc.edu` using your username that you were approved for.
#    1.   You will have to input your password and possibly a 2 step verification. Note, if you are a student
#    using your student id, you will be inputting your current student login password.
# 2. Typing `ls` should show the current directory that you are allowed to work in.
#    1. If you are using the UNCC server, you should see `master_data` and `workspace`
# 3. Jobs can be executed via: `qsub face.sh`
#    1. If you get the error: `qsub:  script is written in DOS/Windows text format` you can resolve this using
#    `dos2unix face.sh`
# 4. `qstat` shows all current jobs while `qstat -u [username]` shows the job for the current user
# 5. If you `cd /[projectname]/log` then use `nano [log file name]` you can view the log output of the server.
# 6. If you would like to code the project locally, and if you are using linux Ubuntu 16.04, then you can mount the file
# system on the server via: `sshfs [username]@hpc.uncc.edu: [local_folder_to_mount_to] -o nonempty`
# 7. If you want to add more videos you can do so without a file transfer application by using: `scp -v ./[path to the video you want to send, you can also use * to send all videos or files in the directory]
# [username]@hpc.uncc.edu:/users/[username]/[path to the folder you want to store the videos]`
# 8. If you need to delete a job, then execute `qdel [JOBID]`
# 9. If you need to view the required modules for your script you can call command `module avail`
