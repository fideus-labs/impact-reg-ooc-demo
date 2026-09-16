#!/usr/bin/env bash
# The organisers' fsleyes views of one method on one subject, the commands of the competition's document as they are:
#   bash pipeline/view.sh [--png] <psoct-in-dmri|dmri-in-psoct> <grey|outline> <M|V> <method>    (pixi run view ...)
# psoct-in-dmri shows DMRI.nii.gz under PSOCT_to_DMRI.nii.gz, dmri-in-psoct shows PSOCT.nii.gz under DMRI_to_PSOCT.nii.gz;
# grey shows both in greyscale, outline draws the carried image's tissue as a red outline. The subject is M or V
# (subject_m, subject_v) and the method the Team_C number (pixi run nifti). With --png, `fsleyes render` writes
# competition/renders/<team>_method_<n>_Subject_<X>_<view>_<style>.png instead of opening a window. TEAM (default C)
# and COMPETITION (default competition) name the team and the organisers' folder.
set -euo pipefail
cd "$(dirname "$0")/.."
png=; if [ "${1:-}" = "--png" ]; then png=1; shift; fi
view=${1:?view: psoct-in-dmri or dmri-in-psoct}; style=${2:?style: grey or outline}
subject=${3:?subject: M or V}; method=${4:?method: the Team_C method number}
case ${subject,,} in m|subject_m) Subj=M ;; v|subject_v) Subj=V ;; *) echo "subject: M or V, not $subject" >&2; exit 2 ;; esac
Team=${TEAM:-C}; Meth=$method
cd "${COMPETITION:-competition}" 2>/dev/null || { echo "${COMPETITION:-competition}/: not here (pixi run nifti <preset> <subject> writes it)" >&2; exit 1; }
data=Data/Subject_$Subj; result=Team_${Team}_method_${Meth}/Subject_$Subj
for f in $data/DMRI.nii.gz $data/PSOCT.nii.gz $result/PSOCT_to_DMRI.nii.gz $result/DMRI_to_PSOCT.nii.gz; do
  [ -e "$f" ] || { echo "$(pwd)/$f: not here (pixi run nifti <preset> subject_${Subj,,} writes it)" >&2; exit 1; }
done
# PSOCT display range is different for Subjects M and V
case $Subj in
  M) psoct_range="0 40"; psoct_in_dmri_range="0 40"; psoct_threshold="15 50" ;;
  V) psoct_range="0 0.7"; psoct_in_dmri_range="0 1"; psoct_threshold="0.3 1.0" ;;
esac
dmri_range="0 0.7"; dmri_threshold="0.3 1.0"
case $view/$style in
  psoct-in-dmri/grey)     args=($data/DMRI.nii.gz --displayRange $dmri_range $result/PSOCT_to_DMRI.nii.gz --displayRange $psoct_in_dmri_range) ;;
  psoct-in-dmri/outline)  args=($data/DMRI.nii.gz --displayRange $dmri_range $result/PSOCT_to_DMRI.nii.gz --overlayType mask --maskColour 1 0 0 --threshold $psoct_threshold --outline --outlineWidth 3) ;;
  dmri-in-psoct/grey)     args=($data/PSOCT.nii.gz --displayRange $psoct_range $result/DMRI_to_PSOCT.nii.gz --displayRange $dmri_range) ;;
  dmri-in-psoct/outline)  args=($data/PSOCT.nii.gz --displayRange $psoct_range $result/DMRI_to_PSOCT.nii.gz --overlayType mask --maskColour 1 0 0 --threshold $dmri_threshold --outline --outlineWidth 3) ;;
  *) echo "view: psoct-in-dmri or dmri-in-psoct; style: grey or outline (not $view/$style)" >&2; exit 2 ;;
esac
if [ -n "$png" ]; then
  mkdir -p renders
  out=renders/Team_${Team}_method_${Meth}_Subject_${Subj}_${view}_${style}.png
  echo "fsleyes render -of $out ${args[*]}"
  fsleyes render -of "$out" --size 1800 700 "${args[@]}"
  echo "-> $(pwd)/$out"
else
  echo "fsleyes ${args[*]}"
  exec fsleyes "${args[@]}"
fi
