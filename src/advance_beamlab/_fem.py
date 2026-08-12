r"""Finite-element head modelling for EEG beamforming, via the New York Head.

MNE-Python computes forward solutions with the boundary element method (BEM):
:func:`mne.make_bem_model` builds nested, closed, homogeneous and isotropic
surfaces (typically scalp, outer skull and inner skull) and
:func:`mne.make_forward_solution` solves the quasi-static problem on them. The
BEM is fast and accurate when the assumptions hold, but by construction it
cannot represent anything the surfaces cannot describe: the cerebrospinal fluid
layer, the split of the skull into compact and spongy bone, the anisotropy of
white matter, or the holes in the skull at the orbits and the auditory meatus.
Those simplifications are known to bias EEG source estimates
:footcite:`HuangEtAl2016`, and the finite element method (FEM), which discretises
the whole head volume into tetrahedra and assigns each one its own conductivity,
is the standard remedy. MNE-Python has no FEM solver.

This module supplies the missing piece in the form that costs the user nothing
to compute: a *precomputed* FEM lead field. The New York Head (ICBM-NY) of Huang,
Parra and Haufe :footcite:`HuangEtAl2016` is a six-tissue (scalp, skull, CSF,
grey matter, white matter, air cavities) finite element model of the ICBM152
non-linear average template, solved once at high resolution and distributed as a
lead field evaluated at 74382 cortical mesh nodes for 231 electrodes of the
10-05 system. :func:`read_ny_head_forward` wraps it as an ordinary
:class:`mne.Forward`, which is all any beamformer in MNE-Python or in this
package needs::

    from advance_beamlab import read_ny_head_forward, make_ny_head_info

    fwd = read_ny_head_forward(resolution="10K")
    info = make_ny_head_info(sfreq=250.0)
    # ... and now make_lcmv, make_mcmv, make_recipsiicos_lcmv and make_abmc
    # all accept ``fwd`` exactly as they accept a BEM forward.

Why this is EEG-only
--------------------
A precomputed template lead field can exist for EEG and cannot exist for MEG,
and the reason is geometric rather than a limitation of this implementation.
EEG electrodes are placed on the scalp by a standardised layout, so "Cz on the
average head" is a well-defined position that is the same for every recording
made with that montage. MEG sensors sit in a fixed helmet while the head sits
wherever the participant put it, so the sensor positions *relative to the brain*
differ for every session and are recorded per session in ``info['dev_head_t']``.
There is no template MEG array to solve for. Computing an MEG FEM forward
therefore requires running a solver (SimNIBS/DUNEuro, FieldTrip-SimBio) on the
individual anatomy and the session's own head position, which is outside what a
downloadable model can provide.

This matters less than it might seem. The tissues the BEM misrepresents are
precisely the ones MEG is least sensitive to: the magnetic field is unaffected
by the radially symmetric part of the volume currents, which is why MEG source
estimates are famously insensitive to skull conductivity while EEG estimates are
not :footcite:`HuangEtAl2016`. BEM is a reasonable model for MEG; for EEG it is
the weak link, and that is the case this module addresses.

Licensing
---------
The New York Head is distributed by its authors under the GNU General Public
License v3. It is therefore **downloaded on demand and never redistributed with
this package**, which is BSD-3-Clause: :func:`fetch_ny_head` retrieves it into
the MNE data directory the first time it is needed, exactly as
:mod:`mne.datasets` handles datasets with their own terms. Using the model
requires citing :footcite:`HuangEtAl2016`.

References
----------
.. footbibliography::
"""

# Authors: Sepehr Shirani <sepehrshirani@gmail.com>, <s.shirani@ucl.ac.uk>
#          Muzhi Wang
# License: BSD-3-Clause

import os
from pathlib import Path

import numpy as np
from mne import Forward, SourceSpaces, create_info
from mne.io.constants import FIFF
from mne.transforms import Transform, apply_trans, get_ras_to_neuromag_trans
from mne.utils import _check_option, _validate_type, logger, verbose

# The single file that carries everything: the lead field, the cortical meshes,
# the electrode names and positions, and the fiducials. Served over plain HTTPS
# from the authors' lab page; ``NYHead.zip`` next to it is the Brainstorm
# template and holds the same lead field in a less convenient form.
NY_HEAD_URL = "https://www.parralab.org/nyhead/sa_nyhead.mat"
NY_HEAD_HASH = "sha256:904fc0baef1783b7bfba9fc73a7f0837265fa303d59c338d14518ed87176b30e"
NY_HEAD_SIZE_MB = 678

# Vertex counts of the meshes stored in the file, used to validate ``resolution``
# and to report sizes before a download-free read.
_RESOLUTIONS = ("1K", "2K", "5K", "10K", "75K")

# The lead field is stored in microvolts per nanoampere-metre; MNE-Python's
# forward operators are in volts per ampere-metre. The two differ by
# 1e-6 V / 1e-9 A m = 1e3. See ``tests/test_fem.py::test_ny_head_units``, which
# pins the resulting gains against an MNE BEM forward computed on fsaverage for
# the same electrodes: both land at tens of V/(A m), as they must for a
# 10 nA m cortical dipole to produce the textbook fraction of a microvolt.
_UV_PER_NAM_TO_V_PER_AM = 1e3

# Vertices 1..37191 of the 74382-node mesh are the left hemisphere and the
# remainder the right; the file stores this as ``in_left``/``in_right`` and the
# split is contiguous, which is what lets the sources be handed to MNE in its
# required left-then-right order simply by sorting.
_N_VERT_TOTAL = 74382
_N_VERT_HEMI = 37191


# The international 10-20 system, as standardised by the American Clinical
# Neurophysiology Society: nineteen scalp electrodes. T3, T4, T5 and T6 were
# renamed T7, T8, P7 and P8 when the 10-10 system superseded it
# :footcite:`OostenveldPraamstra2001`, and the later names are used here.
_TEN_TWENTY = "Fp1 Fp2 F7 F3 Fz F4 F8 T7 C3 Cz C4 T8 P7 P3 Pz P4 P8 O1 O2".split()

# The 10-10 system, which extends the above to 74 positions. The model supplies
# 67 of the 73 named here; see ``ny_head_picks`` for which are absent.
_TEN_TEN = (
    "Nz Fpz Fp1 Fp2 AF7 AF3 AFz AF4 AF8 F9 F7 F5 F3 F1 Fz F2 F4 F6 F8 F10 "
    "FT9 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 FT10 T9 T7 C5 C3 C1 Cz C2 C4 C6 "
    "T8 T10 TP9 TP7 CP5 CP3 CP1 CPz CP2 CP4 CP6 TP8 TP10 P9 P7 P5 P3 P1 Pz P2 "
    "P4 P6 P8 P10 PO7 PO3 POz PO4 PO8 O1 Oz O2 Iz".split()
)


def _ny_head_dir():
    """Return the directory the model is cached in, honouring MNE's config."""
    from mne import get_config

    root = get_config("MNE_DATA", None)
    if root is None:
        root = os.path.join(os.path.expanduser("~"), "mne_data")
    return Path(root) / "NYHead"


@verbose
def fetch_ny_head(path=None, *, force_update=False, verbose=None):
    """Download the New York Head finite element model, or find the cached copy.

    The file is roughly 678 MB and is fetched once. Subsequent calls return the
    cached path without touching the network.

    Parameters
    ----------
    path : path-like | None
        Directory to store the model in. The default (``None``) uses
        ``NYHead`` inside the MNE data directory, i.e. the ``MNE_DATA``
        configuration key if set and ``~/mne_data`` otherwise.
    force_update : bool
        Re-download even if a valid cached copy exists.
    %(verbose)s

    Returns
    -------
    fname : pathlib.Path
        Path to ``sa_nyhead.mat``.

    Notes
    -----
    The model is licensed by its authors under the GPL v3, which is why it is
    downloaded rather than shipped with this BSD-licensed package. Publications
    using it must cite :footcite:`HuangEtAl2016`.

    References
    ----------
    .. footbibliography::
    """
    import pooch

    path = _ny_head_dir() if path is None else Path(path)
    fname = path / "sa_nyhead.mat"
    if fname.is_file():
        if not force_update:
            logger.info(f"Using cached New York Head model at {fname}")
            return fname
        # pooch re-downloads only when the checksum disagrees, so a genuine
        # forced refresh has to remove the file first.
        fname.unlink()

    logger.info(
        f"Downloading the New York Head model ({NY_HEAD_SIZE_MB} MB) to {path}.\n"
        "This model is (C) Huang, Parra and Haufe, distributed under the GPL v3; "
        "see https://www.parralab.org/nyhead/. Please cite Huang et al. (2016), "
        "NeuroImage 140:150-162."
    )
    path.mkdir(parents=True, exist_ok=True)
    # pooch's progress bar needs tqdm, which is not a hard dependency of this
    # package; a 678 MB download in silence looks like a hang, so use it when it
    # is importable and fall back quietly when it is not.
    try:
        import tqdm  # noqa: F401

        progressbar = True
    except ImportError:
        progressbar = False
    pooch.retrieve(
        url=NY_HEAD_URL,
        known_hash=NY_HEAD_HASH,
        fname="sa_nyhead.mat",
        path=str(path),
        progressbar=progressbar,
    )
    return fname


def _open(path):
    """Open the model file, giving an actionable error if h5py is missing."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading the New York Head model requires h5py, because the file is "
            "a MATLAB v7.3 (HDF5) file that scipy.io.loadmat cannot read. "
            "Install it with `pip install h5py`."
        ) from exc
    fname = fetch_ny_head() if path is None else Path(path)
    return h5py.File(fname, "r")


def _matlab_str(h5file, ref):
    """Decode one MATLAB character array, which is stored as uint16 codepoints."""
    return "".join(chr(c) for c in np.asarray(h5file[ref]).ravel())


def _mni_to_head_trans(sa):
    """Build the transform from the model's MNI space (mm) to MNE head space (m).

    The file gives the nasion and the two pre-auricular points in MNI
    millimetres, which is exactly the information MNE uses to define its head
    coordinate frame, so the transform is the standard one rather than an
    approximation: :func:`mne.transforms.get_ras_to_neuromag_trans` puts the
    x axis through the pre-auricular points and the y axis through the nasion.
    Working in head coordinates (and in metres) is what makes the resulting
    forward interchangeable with one from :func:`mne.make_forward_solution`.
    """
    nas, lpa, rpa, _ = np.asarray(sa["naspalparori"]).T / 1000.0
    return get_ras_to_neuromag_trans(nas, lpa, rpa), (nas, lpa, rpa)


def _selected_vertices(sa, resolution):
    """Return the 0-based cortex75K vertex indices for the requested mesh."""
    if resolution == "75K":
        return np.arange(_N_VERT_TOTAL)
    key = f"cortex{resolution}"
    idx = np.asarray(sa[key]["in_from_cortex75K"]).ravel().astype(np.int64) - 1
    # The meshes are strict subsets of the 74382-node one, and MNE requires the
    # sources in ascending vertex order within each hemisphere. Because the left
    # hemisphere occupies the lower half of the index range, one global sort
    # delivers both orderings at once.
    return np.sort(idx)


@verbose
def ny_head_montage(path=None, *, verbose=None):
    """Return the 231-electrode montage of the New York Head.

    Parameters
    ----------
    path : path-like | None
        Path to ``sa_nyhead.mat``; downloaded via :func:`fetch_ny_head` if
        ``None``.
    %(verbose)s

    Returns
    -------
    montage : instance of mne.channels.DigMontage
        Montage in MNE head coordinates, with the nasion and pre-auricular
        points included, so that it can be applied to any recording made with
        (a subset of) the 10-05 layout.

    Notes
    -----
    The names follow the extended 10-05 system, so an ordinary 10-20 or
    64-channel cap is a subset of them and ``info.set_montage`` will match it
    by name.

    Not all 231 are on the scalp vault. The set includes four neck electrodes
    (``Nk1``-``Nk4``, reaching 155 mm below the head-coordinate origin) and a
    band of face and cheek positions (``Exx27``-``Exx34``), because the model
    was built for transcranial stimulation targeting as well as for EEG, and
    those inferior positions matter there. They are genuine scalp positions:
    every electrode lies on the surface returned by :func:`ny_head_scalp` to
    within its mesh resolution. A plot of the montage against the cortex alone
    nevertheless makes them look like stray points floating below the brain.
    Pass ``picks`` to :func:`read_ny_head_forward` to restrict the forward to
    the electrodes you actually recorded.

    References
    ----------
    .. footbibliography::
    """
    from mne.channels import make_dig_montage

    with _open(path) as f:
        sa = f["sa"]
        names = [_matlab_str(f, r) for r in sa["clab_electrodes"][:, 0]]
        # Columns 0:3 are the positions; 3:6 are scalp normals, which a montage
        # does not carry.
        pos_mni = np.asarray(sa["locs_3D"]).T[:, :3] / 1000.0
        trans, (nas, lpa, rpa) = _mni_to_head_trans(sa)

    pos = apply_trans(trans, pos_mni)
    return make_dig_montage(
        ch_pos=dict(zip(names, pos, strict=True)),
        nasion=apply_trans(trans, nas),
        lpa=apply_trans(trans, lpa),
        rpa=apply_trans(trans, rpa),
        coord_frame="head",
    )


@verbose
def ny_head_scalp(path=None, *, verbose=None):
    """Return the scalp surface of the New York Head, in head coordinates.

    The model ships the triangulated head surface its finite element mesh was
    built on. It is supplied here because plotting the electrodes without it is
    actively misleading: drawn against the cortex alone, the montage looks like
    a cloud of points floating around and through the brain, when in fact every
    electrode lies on this surface to within its mesh resolution.

    Parameters
    ----------
    path : path-like | None
        Path to ``sa_nyhead.mat``; downloaded via :func:`fetch_ny_head` if
        ``None``.
    %(verbose)s

    Returns
    -------
    rr : ndarray, shape (1082, 3)
        Vertex positions in metres, in MNE head coordinates. This is the same
        frame as :func:`ny_head_montage` and :func:`read_ny_head_forward`.
    tris : ndarray, shape (2160, 3)
        Triangle definitions, zero-based.

    Notes
    -----
    The surface is coarse (1082 vertices, roughly 10 mm spacing) because it is
    the outer boundary of the volume mesh rather than a rendering surface. It
    extends well below the brain, down to the neck, because the montage does
    too; see :func:`ny_head_montage`.

    See Also
    --------
    ny_head_montage
    read_ny_head_forward
    """
    with _open(path) as f:
        sa = f["sa"]
        trans, _ = _mni_to_head_trans(sa)
        rr = np.asarray(sa["head"]["vc"]).T / 1000.0
        tris = np.asarray(sa["head"]["tri"]).T.astype(np.int64) - 1
    return apply_trans(trans, rr), tris


@verbose
def ny_head_plot_indices(resolution="10K", path=None, *, geodesic=True, verbose=None):
    """Index that spreads a source estimate onto the model's dense mesh.

    A source estimate lives on the mesh ``resolution`` selects, but the surface
    worth looking at is the full 74382-vertex one. Painting the coarse values
    onto the dense mesh and leaving the rest at zero produces a speckled map
    rather than a source distribution, because at ``'5K'`` only one vertex in
    fourteen carries a value. The model ships the fix: an index that gives, for
    every dense vertex, the coarse source that represents it. Applying it is
    nearest-source interpolation, which is what makes the map readable, and it
    is not smoothing: no value is averaged or spread beyond its own cell.

    Parameters
    ----------
    resolution : str
        The mesh the estimate lives on, as passed to
        :func:`read_ny_head_forward`.
    path : path-like | None
        Path to ``sa_nyhead.mat``; downloaded via :func:`fetch_ny_head` if
        ``None``.
    geodesic : bool
        Measure "nearest" along the cortical surface (default) rather than
        through the head. The distinction matters across a sulcal bank, where
        two points can be millimetres apart in space and centimetres apart along
        the sheet.
    %(verbose)s

    Returns
    -------
    idx : ndarray, shape (74382,)
        Index into the source ordering of
        :func:`read_ny_head_forward`, so that ``values[idx]`` is a per-vertex
        array on the dense mesh.

    Notes
    -----
    Median distance from a dense vertex to the source representing it is 2.6 mm
    at ``'5K'``, with a 95th percentile of 5.8 mm, so the interpolation is local.

    See Also
    --------
    read_ny_head_forward
    """
    _check_option("resolution", resolution, _RESOLUTIONS)
    if resolution == "75K":
        return np.arange(_N_VERT_TOTAL)
    key = "in_to_cortex75K_geod" if geodesic else "in_to_cortex75K_eucl"
    with _open(path) as f:
        group = f["sa"][f"cortex{resolution}"]
        # Both are 1-based MATLAB indices, and ``in_to_cortex75K_*`` indexes the
        # file's own (unsorted) vertex order.
        from_75k = np.asarray(group["in_from_cortex75K"]).ravel().astype(np.int64) - 1
        to_dense = np.asarray(group[key]).ravel().astype(np.int64) - 1
    # ``read_ny_head_forward`` sorts the vertices, which is what puts the left
    # hemisphere first as MNE requires, so the file's positions have to be
    # mapped through that same permutation.
    order = np.argsort(from_75k)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank[to_dense]


@verbose
def ny_head_picks(system="10-05", path=None, *, verbose=None):
    """Electrode names of the New York Head belonging to a named montage.

    The model carries 231 electrodes, which is a subset of the 10-05 system
    (Oostenveld and Praamstra extended the 74-position 10-10 system to over 300
    positions :footcite:`OostenveldPraamstra2001`) together with face, cheek,
    neck and fiducial positions the model needs for stimulation targeting. Most
    recordings use far fewer, so this returns the names for a conventional
    montage, to be passed to :func:`read_ny_head_forward` as ``picks``.

    Parameters
    ----------
    system : str
        ``'10-20'``, ``'10-10'``, ``'10-05'`` or ``'all'``.
    path : path-like | None
        Path to ``sa_nyhead.mat``; downloaded via :func:`fetch_ny_head` if
        ``None``.
    %(verbose)s

    Returns
    -------
    names : list of str
        Electrode names present in the model, in the conventional order for
        that system.

    Notes
    -----
    **What each returns, and what is missing.** ``'10-20'`` gives all nineteen
    scalp electrodes of the classic system. ``'10-10'`` gives 67 of the 73 named
    positions: the model has no ``F9``, ``F10``, ``T9``, ``T10``, ``TP9`` or
    ``TP10``, the inferior temporal chain that sits below the standard row.
    ``'10-05'`` gives the 161 electrodes whose names are standard, and ``'all'``
    gives every one of the 231.

    **Nineteen electrodes is not enough for correlated sources on this model,**
    and that is worth stating before anyone reaches for the familiar montage.
    On the correlated pair of :ref:`ex-fem-head-model` the peak error is 38.9 mm
    at 19 electrodes and 0.0 mm at 33 and above, and the 38.9 mm has no spread
    at all over ten noise seeds, so it is a bias of the array rather than a
    noise effect. Over eight further correlated pairs 30 to 40 mm apart the
    median error is 20.5 mm at 19 electrodes with every pair above 10 mm,
    against 0.0 mm at 69. A single source, or an uncorrelated pair, localises
    exactly at every count including 19, so the failure is specific to
    correlated sources rather than general. Dropping from 231 to 161 costs
    nothing measurable.

    **The average reference.** The lead field is supplied in common average
    reference over all 231, exactly (residual 1e-15). Any subset breaks that,
    and not gently: the retained columns miss their own average by a median 18
    to 21 per cent of the column norm whatever the size, because a cap is the
    top of this array rather than a well-distributed sample of it. A random 160
    of the 231 gives 3 per cent where the real 10-05 set of that size gives 21. This
    costs nothing provided the rank is handled, because the offending component
    is exactly what an average-reference projector removes: with a projector the
    beamformer reproduces a by-hand re-referenced lead field to 3e-15 in
    relative power. Use :func:`make_ny_head_info`, which supplies one. Without
    it, and without declaring ``rank=dict(eeg=n-1)``, localisation degrades
    badly and erratically.

    See Also
    --------
    read_ny_head_forward
    make_ny_head_info

    References
    ----------
    .. footbibliography::
    """
    _check_option("system", system, ("10-20", "10-10", "10-05", "all"))
    available = set(ny_head_montage(path=path, verbose=False).ch_names)
    if system == "all":
        return list(ny_head_montage(path=path, verbose=False).ch_names)
    if system == "10-05":
        from mne.channels import make_standard_montage

        # Union rather than intersection with MNE's montage: ``Nz`` is a genuine
        # 10-10 position that MNE files as a fiducial rather than a channel, so
        # intersecting alone would drop it and leave 10-10 not nested inside
        # 10-05.
        standard = set(make_standard_montage("standard_1005").ch_names)
        standard.update(_TEN_TEN)
        return sorted(available & standard)
    wanted = _TEN_TWENTY if system == "10-20" else _TEN_TEN
    return [name for name in wanted if name in available]


@verbose
def make_ny_head_info(sfreq=1000.0, path=None, *, verbose=None):
    """Build a measurement info matching the New York Head electrode array.

    The returned info carries the 231 electrodes in the model's own order, at
    their head-coordinate positions. Importantly, it also carries an average
    reference *projector*. The lead field is supplied in common average
    reference, so modelling data that is referenced differently would use the
    wrong topographies; every inverse in MNE-Python (and every beamformer in
    this package) refuses to run on EEG without that projector for exactly this
    reason.

    Parameters
    ----------
    sfreq : float
        Sampling frequency in Hz to record in the info.
    path : path-like | None
        Path to ``sa_nyhead.mat``; downloaded via :func:`fetch_ny_head` if
        ``None``.
    %(verbose)s

    Returns
    -------
    info : instance of mne.Info
        Info with 231 EEG channels and an average-reference projector.

    See Also
    --------
    read_ny_head_forward
    ny_head_montage
    """
    from mne import EvokedArray

    montage = ny_head_montage(path=path, verbose=False)
    names = list(montage.ch_names)
    info = create_info(names, sfreq=sfreq, ch_types="eeg", verbose=False)
    info.set_montage(montage, verbose=False)
    # ``set_eeg_reference`` is defined on data objects rather than on Info, so
    # the projector is created on a throwaway one-sample Evoked and its info
    # kept. This is cheap and avoids hand-building the projection operator.
    tmp = EvokedArray(np.zeros((len(names), 1)), info, verbose=False)
    tmp.set_eeg_reference("average", projection=True, verbose=False)
    return tmp.info


def _make_src(rr, nn, tris, vertno, hemi_id, subject):
    """Assemble one hemisphere's source space in MNE's on-disk layout."""
    n_vert = len(rr)
    inuse = np.zeros(n_vert, dtype=int)
    inuse[vertno] = 1
    return dict(
        id=hemi_id,
        type="surf",
        np=n_vert,
        ntri=len(tris),
        coord_frame=FIFF.FIFFV_COORD_HEAD,
        rr=rr,
        nn=nn,
        tris=tris,
        nuse=len(vertno),
        inuse=inuse,
        vertno=vertno,
        # The remaining fields are the optional patch/distance information that
        # ``mne.add_source_space_distances`` would fill in. MNE treats ``None``
        # as "not computed", which is the honest value here.
        nuse_tri=0,
        use_tris=None,
        patch_inds=None,
        pinfo=None,
        nearest=None,
        nearest_dist=None,
        dist=None,
        dist_limit=None,
        subject_his_id=subject,
    )


@verbose
def read_ny_head_forward(
    path=None,
    *,
    resolution="10K",
    orientation="normal",
    picks=None,
    verbose=None,
):
    """Read the New York Head FEM lead field as an :class:`mne.Forward`.

    Parameters
    ----------
    path : path-like | None
        Path to ``sa_nyhead.mat``; downloaded via :func:`fetch_ny_head` if
        ``None``.
    resolution : str
        Which of the model's nested cortical meshes to use: ``'1K'``, ``'2K'``,
        ``'5K'``, ``'10K'`` (default) or ``'75K'``. Each is a strict subset of
        the next, so a coarse mesh is a genuine subsampling of the same
        geometry rather than a different model. ``'10K'`` is a good default for
        beamforming, where the scan grid rarely needs to be finer than the
        spatial resolution of EEG; ``'75K'`` reads a 412 MB lead field.
    orientation : str
        ``'normal'`` (default) fixes each source perpendicular to the cortical
        surface, which is the physiologically motivated constraint and the one
        the model's authors provide directly. ``'free'`` returns all three
        Cartesian components per source, letting the beamformer choose the
        orientation.
    picks : list of str | None
        Electrode names to keep, in the order given. ``None`` (default) keeps
        all 231. Note that subsetting breaks the exact average reference of the
        lead field; see Notes.
    %(verbose)s

    Returns
    -------
    fwd : instance of mne.Forward
        Forward solution in head coordinates, directly usable with
        :func:`mne.beamformer.make_lcmv`, :func:`~advance_beamlab.make_mcmv`,
        :func:`~advance_beamlab.make_recipsiicos_lcmv` and
        :func:`~advance_beamlab.make_abmc`.

    Notes
    -----
    **Average reference and rank.** The lead field is supplied in common average
    reference, so its rank is one less than the number of electrodes (230 of 231)
    and the corresponding data must carry an average-reference projector.
    :func:`make_ny_head_info` builds an info that does. Every beamformer here
    resolves the rank through :func:`mne.compute_rank` and MNE's regularised
    pseudoinverse, so the deficiency is handled rather than merely tolerated,
    but it is real and a covariance estimated from these data will be singular.

    If ``picks`` selects a subset of electrodes, the retained columns no longer
    sum to zero across channels, because the average was taken over all 231.
    The subset is still a valid forward model for average-referenced data
    recorded on those electrodes to within the difference between the two
    averages; for a large, well-distributed subset this is small, but it is an
    approximation and not an identity.

    **Visualisation.** The source space is the model's own cortical mesh, not a
    FreeSurfer subject, so ``stc.plot()``, which looks up a subject in a
    ``subjects_dir``, does not apply. The mesh travels with the forward
    (``fwd['src'][hemi]['rr']`` and ``['tris']``), so a
    :class:`~mne.SourceEstimate` can be rendered directly with PyVista; see
    :ref:`ex-fem-head-model` for the dozen lines that do it.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(resolution, str, "resolution")
    _check_option("resolution", resolution, _RESOLUTIONS)
    _check_option("orientation", orientation, ("normal", "free"))

    with _open(path) as f:
        sa = f["sa"]
        names = [_matlab_str(f, r) for r in sa["clab_electrodes"][:, 0]]
        trans, _ = _mni_to_head_trans(sa)
        rot = trans[:3, :3]

        cortex = sa["cortex75K"]
        rr_mni = np.asarray(cortex["vc"]).T / 1000.0
        nn_mni = np.asarray(cortex["normals"]).T
        tris = np.asarray(cortex["tri"]).T.astype(np.int64) - 1

        sel = _selected_vertices(sa, resolution)
        logger.info(
            f"Reading the New York Head lead field: {len(sel)} sources "
            f"(cortex{resolution}), {len(names)} electrodes, "
            f"{orientation} orientation."
        )
        if orientation == "normal":
            # (n_src, n_chan) on disk; the normal projection is precomputed.
            gain = np.asarray(cortex["V_fem_normal"][sel, :]).T
        else:
            # (3, n_src, n_chan) on disk -> (n_chan, n_src, 3).
            gain = np.asarray(cortex["V_fem"][:, sel, :]).transpose(2, 1, 0)

    gain = gain * _UV_PER_NAM_TO_V_PER_AM

    # Head coordinates. Positions are rotated and translated; the free
    # orientation components are a *basis*, so only the rotation applies: a
    # dipole whose moment is m in head coordinates has moment rot.T @ m in the
    # model's frame, hence the lead field transforms as G_head = G_mni @ rot.T.
    rr = apply_trans(trans, rr_mni)
    nn = nn_mni @ rot.T
    if orientation == "free":
        gain = np.einsum("csk,jk->csj", gain, rot).reshape(len(names), -1)

    if picks is not None:
        missing = sorted(set(picks) - set(names))
        if missing:
            raise ValueError(
                f"picks contains {len(missing)} name(s) that are not New York "
                f"Head electrodes, first few: {missing[:5]}"
            )
        keep = [names.index(p) for p in picks]
        gain, names = gain[keep], list(picks)

    # Sources, split into the two hemispheres MNE expects, left first.
    lh_sel = sel[sel < _N_VERT_HEMI]
    rh_sel = sel[sel >= _N_VERT_HEMI] - _N_VERT_HEMI
    lh_tris = tris[np.all(tris < _N_VERT_HEMI, axis=1)]
    rh_tris = tris[np.all(tris >= _N_VERT_HEMI, axis=1)] - _N_VERT_HEMI
    src = SourceSpaces(
        [
            _make_src(
                rr[:_N_VERT_HEMI],
                nn[:_N_VERT_HEMI],
                lh_tris,
                lh_sel,
                FIFF.FIFFV_MNE_SURF_LEFT_HEMI,
                "ICBM-NY",
            ),
            _make_src(
                rr[_N_VERT_HEMI:],
                nn[_N_VERT_HEMI:],
                rh_tris,
                rh_sel,
                FIFF.FIFFV_MNE_SURF_RIGHT_HEMI,
                "ICBM-NY",
            ),
        ]
    )

    n_src = len(sel)
    source_rr = rr[sel]
    if orientation == "normal":
        source_ori = FIFF.FIFFV_MNE_FIXED_ORI
        source_nn = nn[sel]
    else:
        source_ori = FIFF.FIFFV_MNE_FREE_ORI
        # Free orientation: the three columns per source are the head-coordinate
        # axes, which is the convention ``make_forward_solution`` produces.
        source_nn = np.tile(np.eye(3), (n_src, 1))

    info = create_info(names, sfreq=1000.0, ch_types="eeg", verbose=False)
    info.set_montage(ny_head_montage(path=path, verbose=False), verbose=False)
    # A forward's info additionally records where the geometry came from. MNE
    # writes these fields verbatim in ``write_forward_solution``, so without
    # them the forward is usable in memory but cannot be saved to FIF. Saving
    # it is how a user avoids re-reading a 412 MB lead field.
    with info._unlock():
        info["mri_file"] = "ICBM-NY (New York Head)"
        info["mri_id"] = None
        info["meas_file"] = str(NY_HEAD_URL)
        info["mri_head_t"] = Transform("mri", "head", trans)
        info["working_dir"] = ""
        info["command_line"] = "advance_beamlab.read_ny_head_forward"

    fwd = Forward(
        source_ori=source_ori,
        coord_frame=FIFF.FIFFV_COORD_HEAD,
        nsource=n_src,
        nchan=len(names),
        sol=dict(
            data=gain,
            nrow=gain.shape[0],
            ncol=gain.shape[1],
            row_names=list(names),
            col_names=[],
        ),
        _orig_sol=gain.copy(),
        sol_grad=None,
        _orig_sol_grad=None,
        mri_head_t=Transform("mri", "head", trans),
        info=info,
        src=src,
        source_rr=source_rr,
        source_nn=source_nn,
        _orig_source_ori=source_ori,
        surf_ori=orientation == "normal",
    )
    return fwd
