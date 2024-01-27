#ifndef EAMXX_PBL_INVERSION_HEIGHT_HPP
#define EAMXX_PBL_INVERSION_HEIGHT_HPP

#include "share/atm_process/atmosphere_diagnostic.hpp"

namespace scream {

/*
 * This diagnostic will compute the PBL inversion height.
 */

class PBLINversionHeight : public AtmosphereDiagnostic {
 public:
  // Constructors
  PBLINversionHeight(const ekat::Comm &comm, const ekat::ParameterList &params);

  // The name of the diagnostic
  std::string name() const override { return "pbl_inversion_height"; }

  // Set the grid
  void set_grids(
      const std::shared_ptr<const GridsManager> grids_manager) override;

 protected:
#ifdef KOKKOS_ENABLE_CUDA
 public:
#endif
  void compute_diagnostic_impl() override;

  int m_ncols;
  int m_nlevs;
};

}  // namespace scream

#endif  // EAMXX_PBL_INVERSION_HEIGHT_HPP
