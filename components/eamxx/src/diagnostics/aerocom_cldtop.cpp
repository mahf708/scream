#include "diagnostics/aerocom_cldtop.hpp"

#include <ekat/kokkos/ekat_kokkos_utils.hpp>

#include "share/util/scream_common_physics_functions.hpp"

namespace scream {

AeroComCldTop::AeroComCldTop(const ekat::Comm &comm,
                             const ekat::ParameterList &params)
    : AtmosphereDiagnostic(comm, params) {
  // Nothing to do here
}

void AeroComCldTop::set_grids(
    const std::shared_ptr<const GridsManager> grids_manager) {
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  auto grid             = grids_manager->get_grid("Physics");
  const auto &grid_name = grid->name();

  const auto nondim = Units::nondimensional();
  const auto micron = m / 1000000;

  m_ncols = grid->get_num_local_dofs();
  m_nlevs = grid->get_num_vertical_levels();
  m_ndiag = 2;

  // Define layouts we need (both inputs and outputs)
  FieldLayout scalar2d_layout{{COL, LEV}, {m_ncols, m_nlevs}};
  FieldLayout vector1d_layout{{COL, CMP}, {m_ncols, m_ndiag}};

  // The fields required for this diagnostic to be computed
  add_field<Required>("T_mid", scalar2d_layout, K, grid_name);
  add_field<Required>("pseudo_density", scalar2d_layout, Pa, grid_name);
  add_field<Required>("p_mid", scalar2d_layout, Pa, grid_name);
  add_field<Required>("qv", scalar2d_layout, kg / kg, grid_name);
  add_field<Required>("qc", scalar2d_layout, kg / kg, grid_name);
  add_field<Required>("qi", scalar2d_layout, kg / kg, grid_name);
  add_field<Required>("eff_radius_qc", scalar2d_layout, micron, grid_name);
  add_field<Required>("eff_radius_qi", scalar2d_layout, micron, grid_name);
  add_field<Required>("cldfrac_tot", scalar2d_layout, nondim, grid_name);
  add_field<Required>("nc", scalar2d_layout, kg / kg, grid_name);

  // Construct and allocate the aodvis field
  FieldIdentifier fid("AeroComCldTop", vector1d_layout, nondim, grid_name);
  m_diagnostic_output = Field(fid);
  m_diagnostic_output.allocate_view();

  // Self-document the outputs to parse in post-processing
  using stratt_t = std::map<std::string, std::string>;
  auto d         = get_diagnostic();
  auto &metadata =
      d.get_header().get_extra_data<stratt_t>("io: string attributes");
  metadata["blah"]  = "my att 1";
  metadata["blah2"] = "my att 2";
}

// void AeroComCldTop::compute_aerocom_cloudtop(
//     // inputs
//     int ncol, int nlay, const real2d &tmid, const real2d &pmid,
//     const real2d &p_del, const real2d &z_del, const real2d &qc,
//     const real2d &qi, const real2d &rel, const real2d &rei,
//     const real2d &cldfrac_tot, const real2d &nc,
//     // outputs
//     real1d &T_mid_at_cldtop, real1d &p_mid_at_cldtop,
//     real1d &cldfrac_ice_at_cldtop, real1d &cldfrac_liq_at_cldtop,
//     real1d &cldfrac_tot_at_cldtop, real1d &cdnc_at_cldtop,
//     real1d &eff_radius_qc_at_cldtop, real1d &eff_radius_qi_at_cldtop) {
//   /* The goal of this routine is to calculate properties at cloud top
//    * based on the AeroCOM recommendation. See reference for routine
//    * get_subcolumn_mask in rrtmpg, where equation 14 is used for the
//    * maximum-random overlap assumption for subcolumn generation. We use
//    * equation 13, the column counterpart.
//    */
//   // Set outputs to zero
//   Kokkos::deep_copy(T_mid_at_cldtop, 0.0);
//   Kokkos::deep_copy(p_mid_at_cldtop, 0.0);
//   Kokkos::deep_copy(cldfrac_ice_at_cldtop, 0.0);
//   Kokkos::deep_copy(cldfrac_liq_at_cldtop, 0.0);
//   Kokkos::deep_copy(cldfrac_tot_at_cldtop, 0.0);
//   Kokkos::deep_copy(cdnc_at_cldtop, 0.0);
//   Kokkos::deep_copy(eff_radius_qc_at_cldtop, 0.0);
//   Kokkos::deep_copy(eff_radius_qi_at_cldtop, 0.0);
//   // Initialize the 1D "clear fraction" as 1 (totally clear)
//   auto aerocom_clr = real1d("aerocom_clr", ncol);
//   memset(aerocom_clr, 1.0);
//   // Get gravity acceleration constant from constants
//   using physconst = scream::physics::Constants<Real>;
//   // TODO: move tunable constant to namelist
//   constexpr real q_threshold = 0.0;  // BAD_CONSTANT!
//   // TODO: move tunable constant to namelist
//   constexpr real cldfrac_tot_threshold = 0.001;  // BAD_CONSTANT!
//   // Loop over all columns in parallel
//   yakl::fortran::parallel_for(
//       SimpleBounds<1>(ncol), YAKL_LAMBDA(int icol) {
//         // Loop over all layers in serial (due to accumulative
//         // product), starting at 2 (second highest) layer because the
//         // highest is assumed to hav no clouds
//         for(int ilay = 2; ilay <= nlay; ++ilay) {
//           // Only do the calculation if certain conditions are met
//           if((qc(icol, ilay) + qi(icol, ilay)) > q_threshold &&
//              (cldfrac_tot(icol, ilay) > cldfrac_tot_threshold)) {
//             /* PART I: Probabilistically determining cloud top */
//             // Populate aerocom_tmp as the clear-sky fraction
//             // probability of this level, where aerocom_clr is that of
//             // the previous level
//             auto aerocom_tmp =
//                 aerocom_clr(icol) *
//                 (1.0 - ekat::impl::max(cldfrac_tot(icol, ilay - 1),
//                                        cldfrac_tot(icol, ilay))) /
//                 (1.0 - ekat::impl::min(cldfrac_tot(icol, ilay - 1),
//                                        1.0 - cldfrac_tot_threshold));
//             // Temporary variable for probability "weights"
//             auto aerocom_wts = aerocom_clr(icol) - aerocom_tmp;
//             // Temporary variable for liquid "phase"
//             auto aerocom_phi =
//                 qc(icol, ilay) / (qc(icol, ilay) + qi(icol, ilay));
//             /* PART II: The inferred properties */
//             /* In general, converting a 3D property X to a 2D cloud-top
//              * counterpart x follows: x(i) += X(i,k) * weights * Phase
//              * but X and Phase are not always needed */
//             // T_mid_at_cldtop
//             T_mid_at_cldtop(icol) += tmid(icol, ilay) * aerocom_wts;
//             // p_mid_at_cldtop
//             p_mid_at_cldtop(icol) += pmid(icol, ilay) * aerocom_wts;
//             // cldfrac_ice_at_cldtop
//             cldfrac_ice_at_cldtop(icol) += (1.0 - aerocom_phi) * aerocom_wts;
//             // cldfrac_liq_at_cldtop
//             cldfrac_liq_at_cldtop(icol) += aerocom_phi * aerocom_wts;
//             // cdnc_at_cldtop
//             /* We need to convert nc from 1/mass to 1/volume first, and
//              * from grid-mean to in-cloud, but after that, the
//              * calculation follows the general logic */
//             auto cdnc = nc(icol, ilay) * p_del(icol, ilay) / z_del(icol, ilay) /
//                         physconst::gravit / cldfrac_tot(icol, ilay);
//             cdnc_at_cldtop(icol) += cdnc * aerocom_phi * aerocom_wts;
//             // eff_radius_qc_at_cldtop
//             eff_radius_qc_at_cldtop(icol) +=
//                 rel(icol, ilay) * aerocom_phi * aerocom_wts;
//             // eff_radius_qi_at_cldtop
//             eff_radius_qi_at_cldtop(icol) +=
//                 rei(icol, ilay) * (1.0 - aerocom_phi) * aerocom_wts;
//             // Reset aerocom_clr to aerocom_tmp to accumulate
//             aerocom_clr(icol) = aerocom_tmp;
//           }
//         }
//         // After the serial loop over levels, the cloudy fraction is
//         // defined as (1 - aerocom_clr). This is true because
//         // aerocom_clr is the result of accumulative probabilities
//         // (their products)
//         cldfrac_tot_at_cldtop(icol) = 1.0 - aerocom_clr(icol);
//       });
// }

void AeroComCldTop::compute_diagnostic_impl() {
  using KT  = KokkosTypes<DefaultDevice>;
  using MT  = typename KT::MemberType;
  using ESU = ekat::ExeSpaceUtils<typename KT::ExeSpace>;

  using PF = scream::PhysicsFunctions<DefaultDevice>;

  const auto out = m_diagnostic_output.get_view<Real **>();

  // Get the input fields
  const auto tmid = get_field_in("T_mid").get_view<const Real **>();
  const auto pden = get_field_in("pseudo_density").get_view<const Real **>();
  const auto pmid = get_field_in("p_mid").get_view<const Real **>();
  const auto qv   = get_field_in("qv").get_view<const Real **>();
  const auto qc   = get_field_in("qc").get_view<const Real **>();
  const auto qi   = get_field_in("qi").get_view<const Real **>();
  const auto rel  = get_field_in("eff_radius_qc").get_view<const Real **>();
  const auto rei  = get_field_in("eff_radius_qi").get_view<const Real **>();
  const auto cld  = get_field_in("cldfrac_tot").get_view<const Real **>();
  const auto nc   = get_field_in("nc").get_view<const Real **>();

  const auto num_levs = m_nlevs;
  const auto policy   = ESU::get_default_team_policy(m_ncols, m_nlevs);
  Kokkos::parallel_for(
      "Compute " + name(), policy, KOKKOS_LAMBDA(const MT &team) {
        const int icol = team.league_rank();

        auto tmid_icol = ekat::subview(tmid, icol);
        auto pden_icol = ekat::subview(pden, icol);
        auto pmid_icol = ekat::subview(pmid, icol);
        auto qv_icol   = ekat::subview(qv, icol);
        auto qc_icol   = ekat::subview(qc, icol);
        auto qi_icol   = ekat::subview(qi, icol);
        auto rel_icol  = ekat::subview(rel, icol);
        auto rei_icol  = ekat::subview(rei, icol);
        auto cld_icol  = ekat::subview(cld, icol);
        auto nc_icol   = ekat::subview(nc, icol);

        auto dz_icol = PF::calculate_dz(pden_icol,pmid_icol,tmid_icol,qv_icol);

        out(icol, /* test */ 0) = ESU::view_reduction(team, 0, num_levs, qc_icol);
        out(icol, 1) = ESU::view_reduction(team, 0, num_levs, nc_icol);
      });
}

}  // namespace scream
