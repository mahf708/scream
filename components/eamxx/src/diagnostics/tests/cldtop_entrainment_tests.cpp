#include "catch2/catch.hpp"
#include "diagnostics/register_diagnostics.hpp"
#include "share/field/field_utils.hpp"
#include "share/grid/mesh_free_grids_manager.hpp"
#include "share/util/scream_setup_random_test.hpp"

namespace scream {

std::shared_ptr<GridsManager> create_gm(const ekat::Comm &comm, const int ncols,
                                        const int nlevs) {
  const int num_global_cols = ncols * comm.size();

  using vos_t = std::vector<std::string>;
  ekat::ParameterList gm_params;
  gm_params.set("grids_names", vos_t{"Point Grid"});
  auto &pl = gm_params.sublist("Point Grid");
  pl.set<std::string>("type", "point_grid");
  pl.set("aliases", vos_t{"Physics"});
  pl.set<int>("number_of_global_columns", num_global_cols);
  pl.set<int>("number_of_vertical_levels", nlevs);

  auto gm = create_mesh_free_grids_manager(comm, gm_params);
  gm->build_grids();

  return gm;
}

TEST_CASE("cldtop_entrainment") {
  using namespace ShortFieldTagsNames;
  using namespace ekat::units;

  // A world comm
  ekat::Comm comm(MPI_COMM_WORLD);

  // A time stamp
  util::TimeStamp t0({2022, 1, 1}, {0, 0, 0});

  // Create a grids manager - single column for these tests
  constexpr int nlevs = 33;
  const int ngcols    = 2 * comm.size();
  ;
  auto gm   = create_gm(comm, ngcols, nlevs);
  auto grid = gm->get_grid("Physics");

  // Inputs
  auto scalar3d = grid->get_3d_scalar_layout(true);
  FieldIdentifier qc_fid("qc", scalar3d, kg / kg, grid->name());
  FieldIdentifier t_mid_fid("T_mid", scalar3d, K, grid->name());
  FieldIdentifier z_mid_fid("z_mid", scalar3d, m, grid->name());
  Field qc(qc_fid);
  Field t_mid(t_mid_fid);
  Field z_mid(z_mid_fid);
  qc.allocate_view();
  t_mid.allocate_view();
  z_mid.allocate_view();
  qc.get_header().get_tracking().update_time_stamp(t0);
  t_mid.get_header().get_tracking().update_time_stamp(t0);
  z_mid.get_header().get_tracking().update_time_stamp(t0);

  // Construct the Diagnostics
  std::map<std::string, std::shared_ptr<AtmosphereDiagnostic>> diags;
  auto &diag_factory = AtmosphereDiagnosticFactory::instance();
  register_diagnostics();

  // Continue later
}

}  // namespace scream
