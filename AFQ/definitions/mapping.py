import nibabel as nib
import numpy as np
from time import time
import os.path as op

from AFQ.definitions.utils import Definition, find_file
from dipy.align import syn_registration, affine_registration
import AFQ.registration as reg
import AFQ.data as afd

from dipy.align.imaffine import AffineMap

try:
    from fsl.data.image import Image
    from fsl.transform.fnirt import readFnirt
    from fsl.transform.nonlinear import applyDeformation
    has_fslpy = True
except ModuleNotFoundError:
    has_fslpy = False

try:
    import h5py
    has_h5py = True
except ModuleNotFoundError:
    has_h5py = False

__all__ = ["FnirtMap", "SynMap", "SlrMap", "AffMap"]

# For map defintions, get_for_row should return only the mapping
# Where the mapping has transform and transform_inverse functions
# which each accept data, **kwargs


class FnirtMap(Definition):
    """
    Use an existing FNIRT map. Expects a warp file
    and an image file for each subject / session; image file
    is used as src space for warp.

    Parameters
    ----------
    warp_suffix : str
        suffix to pass to bids_layout.get() to identify the warp file.
    space_suffix : str
        suffix to pass to bids_layout.get() to identify the space file.
    warp_filters : str
        Additional filters to pass to bids_layout.get() to identify
        the warp file.
        Default: {}
    space_filters : str
        Additional filters to pass to bids_layout.get() to identify
        the space file.
        Default: {}


    Examples
    --------
    fnirt_map = FnirtMap(
        "warp",
        "MNI",
        {"scope": "TBSS"},
        {"scope": "TBSS"})
    api.AFQ(mapping=fnirt_map)
    """

    def __init__(self, warp_suffix, space_suffix,
                 warp_filters={}, space_filters={}):
        if not has_fslpy:
            raise ImportError(
                "Please install fslpy if you want to use FnirtMap")
        self.warp_suffix = warp_suffix
        self.space_suffix = space_suffix
        self.warp_filters = warp_filters
        self.space_filters = space_filters
        self.fnames = {}

    def find_path(self, bids_layout, from_path, subject, session):
        if session not in self.fnames:
            self.fnames[session] = {}

        nearest_warp = find_file(
            bids_layout, from_path, self.warp_filters, self.warp_suffix,
            session, subject)

        nearest_space = find_file(
            bids_layout, from_path, self.space_filters, self.space_suffix,
            session, subject)

        self.fnames[session][subject] = (nearest_warp, nearest_space)

    def get_for_row(self, afq_object, row):
        nearest_warp, nearest_space = self.fnames[row['ses']][row['subject']]

        our_templ = afq_object.reg_template_img
        subj = Image(row['dwi_file'])
        their_templ = Image(nearest_space)
        warp = readFnirt(nearest_warp, their_templ, subj)

        return ConformedFnirtMapping(warp, our_templ.affine)


class ConformedFnirtMapping():
    """
        ConformedFnirtMapping which matches the generic mapping API.
    """

    def __init__(self, warp, ref_affine):
        self.ref_affine = ref_affine
        self.warp = warp

    def transform_inverse(self, data, **kwargs):
        data_img = Image(nib.Nifti1Image(
            data.astype(np.float32), self.ref_affine))
        xform_data = np.asarray(applyDeformation(data_img, self.warp).data)
        return xform_data

    def transform(self, data, **kwargs):
        raise NotImplementedError(
            "Fnirt based mappings can currently"
            + " only transform from template to subject space")


class ItkMap(Definition):
    """
    Use an existing Itk map (e.g., from ANTS). Expects the warp file
    from MNI to T1.

    Parameters
    ----------
    warp_suffix : str
        suffix to pass to bids_layout.get() to identify the warp file.
    warp_filters : str
        Additional filters to pass to bids_layout.get() to identify
        the warp file.
        Default: {}


    Examples
    --------
    itk_map = ItkMap(
        "xfm",
        {"scope": "qsiprep",
        "from": "MNI152NLin2009cAsym",
        "to": "T1w"})
    api.AFQ(mapping=itk_map)
    """

    def __init__(self, warp_suffix, warp_filters={}):
        if not has_h5py:
            raise ImportError(
                "Please install h5py if you want to use ItkMap")
        self.warp_suffix = warp_suffix
        self.warp_filters = warp_filters
        self.fnames = {}

    def find_path(self, bids_layout, from_path, subject, session):
        if session not in self.fnames:
            self.fnames[session] = {}

        self.fnames[session][subject] = find_file(
            bids_layout, from_path, self.warp_filters, self.warp_suffix,
            session, subject, extension="h5")

    def get_for_row(self, afq_object, row):
        nearest_warp = self.fnames[row['ses']][row['subject']]
        warp_f5 = h5py.File(nearest_warp)
        their_shape = np.asarray(warp_f5["TransformGroup"]['1'][
            'TransformFixedParameters'], dtype=int)[:3]
        our_shape = afq_object.reg_template_img.get_fdata().shape
        if (our_shape != their_shape).any():
            raise ValueError((
                f"The shape of your ITK mapping ({their_shape})"
                f" is not the same as your template for registration"
                f" ({our_shape})"))
        their_forward = np.asarray(warp_f5["TransformGroup"]['1'][
            'TransformParameters']).reshape([*their_shape, 3])
        their_disp = np.zeros((*their_shape, 3, 2))
        their_disp[..., 0] = their_forward
        their_disp = nib.Nifti1Image(
            their_disp, afq_object.reg_template_img.affine)
        their_prealign = np.zeros((4, 4))
        their_prealign[:3, :3] = np.asarray(warp_f5["TransformGroup"]["2"][
            "TransformParameters"])[:9].reshape((3, 3))
        their_prealign[:3, 3] = np.asarray(warp_f5["TransformGroup"]["2"][
            "TransformParameters"])[9:]
        their_prealign[3, 3] = 1.0
        warp_f5.close()
        return reg.read_mapping(
            their_disp, row['dwi_file'],
            afq_object.reg_template_img, prealign=their_prealign)


class GeneratedMapMixin(object):
    """
    Helper Class
    Useful for maps that are generated by pyAFQ
    """

    def get_fnames(self, extension, afq_object, row):
        mapping_file = afq_object._get_fname(
            row,
            '_mapping_from-DWI_to_MNI_xfm')
        meta_fname = afq_object._get_fname(row, '_mapping_reg')
        mapping_file = mapping_file + extension
        meta_fname = meta_fname + '.json'
        return mapping_file, meta_fname

    def prealign(self, afq_object, row):
        prealign_file = afq_object._get_fname(
            row, '_prealign_from-DWI_to-MNI_xfm.npy')
        if not op.exists(prealign_file):
            reg_subject_img, _ = afq_object._reg_img(
                afq_object.reg_subject, True, row)
            start_time = time()
            _, aff = affine_registration(
                reg_subject_img,
                afq_object.reg_template_img,
                **self.affine_kwargs)
            np.save(prealign_file, aff)
            meta_fname = afq_object._get_fname(
                row, '_prealign_from-DWI_to-MNI_xfm.json')
            meta = dict(type="rigid")
            afd.write_json(meta_fname, meta)
            row['timing']['Registration_pre_align'] =\
                row['timing']['Registration_pre_align'] + time() - start_time
        return prealign_file

    def get_for_row(self, afq_object, row):
        mapping_file, meta_fname = self.get_fnames(
            self.extension, afq_object, row)

        if self.use_prealign:
            reg_prealign = np.load(self.prealign(afq_object, row))
        else:
            reg_prealign = None
        if not op.exists(mapping_file):
            reg_template_img, reg_template_sls = \
                afq_object._reg_img(afq_object.reg_template, False, row)
            reg_subject_img, reg_subject_sls = \
                afq_object._reg_img(afq_object.reg_subject, True, row)

            start_time = time()
            mapping = self.gen_mapping(
                afq_object, row, reg_template_img, reg_template_sls,
                reg_subject_img, reg_subject_sls, reg_prealign)
            row['timing']['Registration'] =\
                row['timing']['Registration'] + time() - start_time

            reg.write_mapping(mapping, mapping_file)
            meta = dict(type="displacementfield")
            afd.write_json(meta_fname, meta)
        if self.use_prealign:
            reg_prealign_inv = np.linalg.inv(reg_prealign)
        else:
            reg_prealign_inv = None
        mapping = reg.read_mapping(
            mapping_file,
            row['dwi_file'],
            afq_object.reg_template_img,
            prealign=reg_prealign_inv)
        return mapping


class SynMap(GeneratedMapMixin, Definition):
    """
    Calculate a Syn registration for each subject/session
    using reg_subject and reg_template.

    Parameters
    ----------
    use_prealign : bool
        Whether to perform a linear pre-registration.
        Default: True
    affine_kwargs : dictionary, optional
        Parameters to pass to affine_registration
        in dipy.align, which does the linear pre-alignment.
        Only used if use_prealign is True.
        Default: {}
    syn_kwargs : dictionary, optional
        Parameters to pass to syn_registration
        in dipy.align, which does the SyN alignment.
        Default: {}
    Examples
    --------
    api.AFQ(mapping=SynMap())
    """

    def __init__(self, use_prealign=True, affine_kwargs={}, syn_kwargs={}):
        self.use_prealign = use_prealign
        self.affine_kwargs = affine_kwargs
        self.syn_kwargs = syn_kwargs
        self.extension = ".nii.gz"

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def gen_mapping(self, afq_object, row, reg_template_img, reg_template_sls,
                    reg_subject_img, reg_subject_sls, reg_prealign):
        _, mapping = syn_registration(
            reg_subject_img.get_fdata(),
            reg_template_img.get_fdata(),
            moving_affine=reg_subject_img.affine,
            static_affine=reg_template_img.affine,
            prealign=reg_prealign,
            **self.syn_kwargs)
        if self.use_prealign:
            mapping.codomain_world2grid = np.linalg.inv(reg_prealign)
        return mapping


class SlrMap(GeneratedMapMixin, Definition):
    """
    Calculate a SLR registration for each subject/session
    using reg_subject and reg_template.

    syn_kwargs : dictionary, optional
        Parameters to pass to whole_brain_slr
        in dipy, which does the SLR alignment.
        Default: {}

    Examples
    --------
    api.AFQ(mapping=SlrMap())
    """

    def __init__(self, syn_kwargs={}):
        self.syn_kwargs = {}
        self.use_prealign = False
        self.extension = ".npy"

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def gen_mapping(self, afq_object, row, reg_template_img, reg_template_sls,
                    reg_subject_img, reg_subject_sls, reg_prealign):
        return reg.slr_registration(
            reg_subject_sls, reg_template_sls,
            moving_affine=reg_subject_img.affine,
            moving_shape=reg_subject_img.shape,
            static_affine=reg_template_img.affine,
            static_shape=reg_template_img.shape,
            **self.slr_kwargs)


class AffMap(GeneratedMapMixin, Definition):
    """
    Calculate an affine registration for each subject/session
    using reg_subject and reg_template.

    affine_kwargs : dictionary, optional
        Parameters to pass to affine_registration
        in dipy.align, which does the linear pre-alignment.
        Default: {}

    Examples
    --------
    api.AFQ(mapping=AffMap())
    """

    def __init__(self, affine_kwargs={}):
        self.use_prealign = False
        self.affine_kwargs = affine_kwargs
        self.extension = ".npy"

    def find_path(self, bids_layout, from_path, subject, session):
        pass

    def gen_mapping(self, afq_object, row, reg_template_img, reg_template_sls,
                    reg_subject_img, reg_subject_sls, reg_prealign):
        return ConformedAffineMapping(np.load(self.prealign(afq_object, row)))


class ConformedAffineMapping(AffineMap):
    """
    Modifies AffineMap API to match DiffeomorphicMap API.
    Important for SLR maps API to be indistinguishable from SYN maps API.
    """

    def transform(self, *args, interpolation='linear', **kwargs):
        kwargs['interp'] = interpolation
        return super().transform_inverse(*args, **kwargs)

    def transform_inverse(self, *args, interpolation='linear', **kwargs):
        kwargs['interp'] = interpolation
        return super().transform(*args, **kwargs)
