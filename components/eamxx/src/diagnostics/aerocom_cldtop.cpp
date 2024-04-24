#include "diagnostics/aerocom_cldtop.hpp"

#include <ekat/kokkos/ekat_kokkos_utils.hpp>

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

  m_ncols = grid->get_num_local_dofs();
  m_nlevs = grid->get_num_vertical_levels();
  m_ndiag = 2;

  // Define layouts we need (both inputs and outputs)
  FieldLayout scalar2d_layout{{COL, LEV}, {m_ncols, m_nlevs}};
  FieldLayout vector1d_layout{{COL, CMP}, {m_ncols, m_ndiag}};

  // The fields required for this diagnostic to be computed
  add_field<Required>("nc", scalar2d_layout, kg / kg, grid_name);
  add_field<Required>("qc", scalar2d_layout, kg / kg, grid_name);

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

void AeroComCldTop::compute_diagnostic_impl() {
  using KT  = KokkosTypes<DefaultDevice>;
  using MT  = typename KT::MemberType;
  using ESU = ekat::ExeSpaceUtils<typename KT::ExeSpace>;

  const auto out = m_diagnostic_output.get_view<Real **>();
  const auto qc  = get_field_in("qc").get_view<const Real **>();
  const auto nc  = get_field_in("nc").get_view<const Real **>();

  const auto num_levs = m_nlevs;
  const auto policy   = ESU::get_default_team_policy(m_ncols, m_nlevs);
  Kokkos::parallel_for(
      "Compute " + name(), policy, KOKKOS_LAMBDA(const MT &team) {
        const int icol = team.league_rank();
        auto qc_icol   = ekat::subview(qc, icol);
        auto nc_icol   = ekat::subview(nc, icol);
        out(icol, 0)   = ESU::view_reduction(team, 0, num_levs, qc_icol);
        out(icol, 1)   = ESU::view_reduction(team, 0, num_levs, nc_icol);
      });
}

}  // namespace scream
