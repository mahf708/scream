#include "diagnostics/cldtop_entrainment.hpp"

#include <ekat/kokkos/ekat_kokkos_utils.hpp>

namespace scream {

Cldtop_Entrainment::Cldtop_Entrainment(const ekat::Comm &comm,
                                       const ekat::ParameterList &params)
    : AtmosphereDiagnostic(comm, params) {
  // Nothing to do here
}

void Cldtop_Entrainment::set_grids(
    const std::shared_ptr<const GridsManager> grids_manager) {
  using namespace ekat::units;
  using namespace ShortFieldTagsNames;

  auto grid             = grids_manager->get_grid("Physics");
  const auto &grid_name = grid->name();

  const auto nondim = Units::nondimensional();

  m_ncols = grid->get_num_local_dofs();
  m_nlevs = grid->get_num_vertical_levels();

  // Define layouts we need (both inputs and outputs)
  auto scalar3d = grid->get_3d_scalar_layout(true);
  auto vector2d = grid->get_2d_vector_layout(true, CMP, m_num_outputs);

  // The fields required for this diagnostic to be computed
  add_field<Required>("qc", scalar3d, kg / kg, grid_name);
  add_field<Required>("T_mid", scalar3d, K, grid_name);
  add_field<Required>("z_mid", vector2d, m, grid_name);

  // Construct and allocate the cldtop_entrainment field
  // We are going to assume we have nondim units here for ease
  FieldIdentifier fid("cldtop_entrainment", vector2d, nondim, grid_name);
  m_diagnostic_output = Field(fid);
  m_diagnostic_output.allocate_view();
}

void Cldtop_Entrainment::compute_diagnostic_impl() {
  using KT  = KokkosTypes<DefaultDevice>;
  using MT  = typename KT::MemberType;
  using ESU = ekat::ExeSpaceUtils<typename KT::ExeSpace>;

  const auto &qc    = get_field_in("qc").get_view<const Real ***>();
  const auto &t_mid = get_field_in("T_mid").get_view<const Real ***>();
  const auto &z_mid = get_field_in("z_mid").get_view<const Real ***>();

  // Note that cte has dimensions of (ncol,m_num_outputs)
  // m_num_outputs is defined in cltop_entrainment.hpp
  const auto &cte = m_diagnostic_output.get_view<Real **>();

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
            cte(icol, 0) = z_mid(icol, ilay);
            break;
          }
        }
        team.team_barrier();
      });
}

}  // namespace scream
