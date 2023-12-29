import shutil
from dataclasses import dataclass
from typing import Union, Optional, Tuple, List, Dict

import numpy as np
import os
from pathlib import Path
import nibabel as nib
from tqdm import tqdm


@dataclass
class SubjectFiles:
    name: str = None
    sax: List[str] = None
    la4ch: str = None
    la3ch: str = None
    la2ch: str = None
    sax_seg: List[str] = None
    la4ch_seg: str = None
    la3ch_seg: str = None
    la2ch_seg: str = None


def split_sax_into_slices(sax_path: str, save_dir: str, skip_exist=True) -> Tuple[List[str], List[str]]:
    sax_path = Path(sax_path)
    assert sax_path.exists()
    assert sax_path.is_file()
    assert sax_path.name == "sa.nii.gz"
    seg_sax_path = sax_path.parent / "seg_sa.nii.gz"
    assert seg_sax_path.exists()
    assert seg_sax_path.is_file()
    subject_save_dir = Path(save_dir)
    subject_save_dir.parent.mkdir(exist_ok=True)
    subject_save_dir.mkdir(exist_ok=True)

    # If we assume the files have been processed correectly in the past, we can skip the process
    if skip_exist:
        files = [str(i) for i in subject_save_dir.iterdir() if i.is_file() and str(i.name)[:3] != "seg" and str(i.name)[-7:] == ".nii.gz"]
        seg_files = [str(i) for i in subject_save_dir.iterdir() if i.is_file() and str(i.name)[:3] == "seg" and str(i.name)[-7:] == ".nii.gz"]
        if files:
            expected_total = int(files[0].split("-")[-1].split(".")[0])
            if len(files) == expected_total:
                assert len(files) == len(seg_files)
                return files, seg_files

    # Load volumes
    sax_nii = nib.load(str(sax_path))
    sax_im = sax_nii.dataobj[:]
    seg_sax_nii = nib.load(str(seg_sax_path))
    seg_sax_im = seg_sax_nii.dataobj[:]

    # Retrive affine matrix and get origin of volume in scanner space
    sax_aff = sax_nii.affine
    voxel_origin = np.array([0, 0, 0, 0])
    scanner_origin = sax_aff @ voxel_origin

    slice_files = []
    seg_slice_files = []
    for slice_idx in range(sax_im.shape[2]):
        # Get the position of this slice's origin in scanner space
        z_from_origin = slice_idx
        slice_origin = np.array([voxel_origin[0], voxel_origin[1], z_from_origin, 0])
        # Calculate the distance in scanner space of this slice from volume's origin
        dist_from_vol_origin = sax_aff @ slice_origin - scanner_origin

        # The last column of the affine matrix is the inficates the displacement of the volume from the scanner origin
        # Adding the volume_origin -> slice_origin distance to the affine's offset gives us the slice's affine
        slice_affine = sax_aff.copy()
        slice_affine[:3, 3] += dist_from_vol_origin[:3]

        # Save this image slice individually
        slice_nii = nib.Nifti1Image(sax_im[:, :, slice_idx:slice_idx+1, :], slice_affine)
        slice_path = str(subject_save_dir / f"sa_{slice_idx}-{sax_im.shape[2]}.nii.gz")
        nib.save(slice_nii, slice_path)

        # Save this segmentation slice individually
        seg_slice_nii = nib.Nifti1Image(seg_sax_im[:, :, slice_idx:slice_idx+1, :], slice_affine)
        seg_slice_path = str(subject_save_dir / f"seg_sa_{slice_idx}-{sax_im.shape[2]}.nii.gz")
        nib.save(seg_slice_nii, seg_slice_path)

        slice_files.append(slice_path)
        seg_slice_files.append(seg_slice_path)
    assert len(slice_files) == len(seg_slice_files)
    return slice_files, seg_slice_files


def find_subjects(dataset_dir: str, sax_slice_dataset_dir: str,
                  remove_sa_slices: bool = False, num_subj: int = None) -> List[SubjectFiles]:
    dataset_dir = Path(dataset_dir)
    sax_slice_dataset_dir = Path(sax_slice_dataset_dir)
    assert dataset_dir.is_dir()
    subject_list = []
    subjects = list(dataset_dir.iterdir())
    if num_subj is not None:
        assert num_subj > 0
        print(f"Only {num_subj}/{len(subjects)} will be procesed")
        subjects = subjects[:num_subj]

    for subj_dir in tqdm(subjects, desc="Iterating over subject directory"):
        sax_slices = None
        seg_sax_slices = None
        if (subj_dir / "sa_slices").exists():
            sa_files = list((subj_dir / "sa_slices").iterdir())
            sax_slices = [str(i) for i in sa_files if i.name[:2] == "sa"]
            seg_sax_slices = [str(i) for i in sa_files if i.name[:3] == "seg"]
        if sax_slices is None or not sax_slices or remove_sa_slices:
            if remove_sa_slices:
                try:
                    shutil.rmtree(str(subj_dir / "sa_slices"))
                except FileNotFoundError:
                    pass
            sa_file = subj_dir / "sa.nii.gz"
            if not sa_file.exists() or not sa_file.is_file() or sa_file.stat().st_size == 0:
                continue
            try:
                sax_slices, seg_sax_slices = split_sax_into_slices(str(sa_file), save_dir=str(sax_slice_dataset_dir / subj_dir.name / "sa_slices"))
            except Exception as e:
                continue
        la4ch = subj_dir / "la_4ch.nii.gz"
        if not la4ch.exists() or not la4ch.is_file() or la4ch.stat().st_size == 0:
            continue
        la3ch = subj_dir / "la_3ch.nii.gz"
        if not la3ch.exists() or not la3ch.is_file() or la3ch.stat().st_size == 0:
            continue
        la2ch = subj_dir / "la_2ch.nii.gz"
        if not la2ch.exists() or not la2ch.is_file() or la2ch.stat().st_size == 0:
            continue
        seg_la4ch = subj_dir / "seg_la_4ch.nii.gz"
        if not seg_la4ch.exists() or not seg_la4ch.is_file() or seg_la4ch.stat().st_size == 0:
            continue
        seg_la3ch = subj_dir / "seg_la_3ch.nii.gz"
        # if not seg_la3ch.exists() or not seg_la3ch.is_file() or seg_la3ch.stat().st_size == 0:
        #     continue
        seg_la2ch = subj_dir / "seg_la_2ch.nii.gz"
        if not seg_la2ch.exists() or not seg_la2ch.is_file() or seg_la2ch.stat().st_size == 0:
            continue

        subj_files = SubjectFiles(name=subj_dir.name, sax=sax_slices, la4ch=str(la4ch), la3ch=str(la3ch), la2ch=str(la2ch), sax_seg=seg_sax_slices, la4ch_seg=str(seg_la4ch), la3ch_seg=str(seg_la3ch), la2ch_seg=str(seg_la2ch))
        subject_list.append(subj_files)
    assert subject_list
    return subject_list


def download_subject_images(subject_list: List[str],
                            download_dir: str,
                            hostname="-",
                            username="-",
                            password=":)",
                            subjects_folder="/vol/aimspace/projects/ukbb/cardiac/cardiac_segmentations/subjects/",
                            remove_subj_on_issue=True,
                            max_download_num=-1,
                            ):
    import paramiko
    # Create SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=username, password=password)
    # Create SCP client
    scp = ssh.open_sftp()

    Path(download_dir).mkdir(exist_ok=True)
    temp_dir = Path(download_dir) / "temp"
    temp_dir.mkdir(exist_ok=True)
    count = 0

    # Iterate over remote subdirectories and download matching ones
    for i, subject_id in tqdm(list(enumerate(subject_list)), desc="Downloading subject series..."):
        subject_subdir = os.path.join(subjects_folder, subject_id+"/")
        stdin, stdout, stderr = ssh.exec_command(f'test -d {subject_subdir} && echo 1 || echo 0')
        output = stdout.read().decode().strip()
        if output == '0':
            print(f'The subject directory does not exist: {subject_subdir}')
            continue
        local_file_path = Path(temp_dir) / subject_id
        local_file_path.mkdir(exist_ok=True)
        final_file_path = Path(download_dir) / subject_id

        # SAX download
        remote_file_path = os.path.join(subject_subdir, "sa.nii.gz")
        local_file_path_sax = local_file_path / "sa.nii.gz"
        if not local_file_path_sax.exists():
            try:
                scp.get(remote_file_path, str(local_file_path_sax))
            except FileNotFoundError as e:
                print(f"Subject SAX series not found: {remote_file_path}")
                # scp.get might create an empty file if a FileNotFoundError is found
                if local_file_path_sax.exists():
                    os.remove(str(local_file_path_sax))
                if remove_subj_on_issue:
                    shutil.rmtree(local_file_path)
                    continue

        # SAX segmentation download
        remote_file_path = os.path.join(subject_subdir, "seg_sa.nii.gz")
        local_file_path_sax = local_file_path / "seg_sa.nii.gz"
        if not local_file_path_sax.exists():
            try:
                scp.get(remote_file_path, str(local_file_path_sax))
            except FileNotFoundError as e:
                print(f"Subject SAX segmentation not found: {remote_file_path}")
                # scp.get might create an empty file if a FileNotFoundError is found
                if local_file_path_sax.exists():
                    os.remove(str(local_file_path_sax))
                if remove_subj_on_issue:
                    shutil.rmtree(local_file_path)
                    continue

        # 4CH download
        remote_file_path = os.path.join(subject_subdir, "la_4ch.nii.gz")
        local_file_path_4ch = local_file_path / "la_4ch.nii.gz"
        if not local_file_path_4ch.exists():
            try:
                scp.get(remote_file_path, str(local_file_path_4ch))
            except FileNotFoundError as e:
                print(f"Subject 4CH series not found: {remote_file_path}")
                # scp.get might create an empty file if a FileNotFoundError is found
                if local_file_path_4ch.exists():
                    os.remove(str(local_file_path_4ch))
                if remove_subj_on_issue:
                    shutil.rmtree(local_file_path)
                    continue

        # 4CH segmentation download
        remote_file_path = os.path.join(subject_subdir, "seg_la_4ch.nii.gz")
        local_file_path_4ch = local_file_path / "seg_la_4ch.nii.gz"
        if not local_file_path_4ch.exists():
            try:
                scp.get(remote_file_path, str(local_file_path_4ch))
            except FileNotFoundError as e:
                print(f"Subject 4CH segmentation not found: {remote_file_path}")
                # scp.get might create an empty file if a FileNotFoundError is found
                if local_file_path_4ch.exists():
                    os.remove(str(local_file_path_4ch))
                if remove_subj_on_issue:
                    shutil.rmtree(local_file_path)
                    continue

        # 3CH download
        remote_file_path = os.path.join(subject_subdir, "la_3ch.nii.gz")
        local_file_path_3ch = local_file_path / "la_3ch.nii.gz"
        if not local_file_path_3ch.exists():
            try:
                scp.get(remote_file_path, str(local_file_path_3ch))
            except FileNotFoundError as e:
                print(f"Subject 3CH series not found: {remote_file_path}")
                # scp.get might create an empty file if a FileNotFoundError is found
                if local_file_path_3ch.exists():
                    os.remove(str(local_file_path_3ch))
                if remove_subj_on_issue:
                    shutil.rmtree(local_file_path)
                    continue
        # There is no 3CH segmentation

        # 2CH download
        remote_file_path = os.path.join(subject_subdir, "la_2ch.nii.gz")
        local_file_path_2ch = local_file_path / "la_2ch.nii.gz"
        if not local_file_path_2ch.exists():
            try:
                scp.get(remote_file_path, str(local_file_path_2ch))
            except FileNotFoundError as e:
                print(f"Subject 2CH series not found: {remote_file_path}")
                # scp.get might create an empty file if a FileNotFoundError is found
                if local_file_path_2ch.exists():
                    os.remove(str(local_file_path_2ch))
                if remove_subj_on_issue:
                    shutil.rmtree(local_file_path)
                    continue

        # 2CH segmentation download
        remote_file_path = os.path.join(subject_subdir, "seg_la_2ch.nii.gz")
        local_file_path_2ch = local_file_path / "seg_la_2ch.nii.gz"
        if not local_file_path_2ch.exists():
            try:
                scp.get(remote_file_path, str(local_file_path_2ch))
            except FileNotFoundError as e:
                print(f"Subject 2CH segmentation not found: {remote_file_path}")
                # scp.get might create an empty file if a FileNotFoundError is found
                if local_file_path_2ch.exists():
                    os.remove(str(local_file_path_2ch))
                if remove_subj_on_issue:
                    shutil.rmtree(local_file_path)
                    continue
        os.rename(str(local_file_path), str(final_file_path))
        count += 1
        if count % 200 == 0:
            print(f"Count: {count}/{i}  ({count/i}%)")
        if max_download_num > 0 and count == max_download_num:
            break
    print(f"Downloaded {max_download_num} subjects.")
    # Close SCP and SSH clients
    scp.close()
    ssh.close()


def download_substring_matching_subjects(keys: List[str],
                                         download_dir: str,
                                         hostname="-",
                                         username="-",
                                         password=":)",
                                         subjects_folder="/vol/aimspace/projects/ukbb/cardiac/cardiac_segmentations/subjects/",
                                         ):
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    ssh.connect(hostname=hostname, username=username, password=password)
    sftp = ssh.open_sftp()
    subjects = []
    for entry in sftp.listdir_attr(subjects_folder):
        if entry.filename.startswith('.'):
            continue
        if keys:
            for k in keys:
                if entry.filename[:len(k)] == k:
                    subjects.append(entry.filename)
                    break
        else:
            subjects.append(entry.filename)

    sftp.close()
    ssh.close()
    download_subject_images(subjects, download_dir, hostname=hostname, username=username, password=password, subjects_folder=subjects_folder)


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "1"
    remote_subj_dir = "/vol/aimspace/projects/ukbb/cardiac/cardiac_segmentations/subjects/"
    download_dir = "/home/pti/Documents/datasets/UKBB_subjects/"
    hostname = input("Hostname:")
    username = input("Username:")
    password = input("Password:")
    download_substring_matching_subjects([], download_dir, subjects_folder=remote_subj_dir, password=password)
    subject_list = find_subjects(download_dir, sax_slice_dataset_dir="/home/pti/Documents/datasets/UKBB_subjects_unaligned")
    print(len(subject_list))
    # ims_, anns_ = [i for i, j, in ann_pairs], [j for i, j, in ann_pairs]
    # print(len(ann_pairs), "SAX subjects")
