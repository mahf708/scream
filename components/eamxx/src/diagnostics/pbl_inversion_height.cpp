#include "diagnostics/pbl_inversion_height.hpp"

#include <ekat/kokkos/ekat_kokkos_utils.hpp>

namespace scream {

PBLInversionHeight::PBLInversionHeight(const ekat::Comm &comm,
                                       const ekat::ParameterList &params)
    : AtmosphereDiagnostic(comm, params) {
  // Nothing to do here
}

void PBLInversionHeight::set_grids(
    const std::shared_ptr<const GridsManager> grids_manager) {
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  auto grid             = grids_manager->get_grid("Physics");
  const auto &grid_name = grid->name();

  m_ncols = grid->get_num_local_dofs();
  m_nlevs = grid->get_num_vertical_levels();

  // Define layouts we need (both inputs and outputs)
  auto scalar3d = grid->get_3d_scalar_layout(true);
  auto scalar2d = grid->get_2d_scalar_layout(true);

  // The fields required for this diagnostic to be computed
  add_field<Required>("qc", scalar3d, kg / kg, grid_name);
  add_field<Required>("T_mid", scalar3d, K, grid_name);
  add_field<Required>("z_mid", scalar3d, m, grid_name);

  // Construct and allocate the pbl_inversion_height field
  FieldIdentifier fid("pbl_inversion_height", scalar2d, m, grid_name);
  m_diagnostic_output = Field(fid);
  m_diagnostic_output.allocate_view();
}

void PBLINversionHeight::compute_diagnostic_impl() {
  using KT  = KokkosTypes<DefaultDevice>;
  using MT  = typename KT::MemberType;
  using ESU = ekat::ExeSpaceUtils<typename KT::ExeSpace>;

  const auto &qc    = get_field_in("qc").get_view<const Real ***>();
  const auto &t_mid = get_field_in("T_mid").get_view<const Real ***>();
  const auto &z_mid = get_field_in("z_mid").get_view<const Real ***>();

  const auto &pih = m_diagnostic_output.get_view<Real **>();

  const auto num_levs = m_num_levs;
  const auto policy   = ESU::get_default_team_policy(m_num_cols, m_num_levs);
  Kokkos::parallel_for(
      "Compute " + name(), policy, KOKKOS_LAMBDA(const MT &team) {
        const int icol  = team.league_rank();
        auto qc_icol    = ekat::subview(q, icol);
        auto t_mid_icol = ekat::subview(t_mid, icol);
        // calculate the first height where qc(lev) > qc(lev-1)
        for(int ilay = num_levs; ilay >= 0; --ilay) {
          // TODO: Or is it the other way around?
          // TODO: Also add other methods here
          if(qc_icol(icol, ilay) > qc_icol(icol, ilay - 1)) {
            pih(icol) = z_mid(icol, ilay);
            break;
          }
        }
        team.team_barrier();
      });
}

}  // namespace scream
