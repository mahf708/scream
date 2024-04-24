#ifndef EAMXX_AEROCOMCLDTOP_DIAG
#define EAMXX_AEROCOMCLDTOP_DIAG

#include "share/atm_process/atmosphere_diagnostic.hpp"

namespace scream {

/*
 * This diagnostic will compute the AeroCom diagnostics.
 */

class AeroComCldTop : public AtmosphereDiagnostic {
 public:
  // Constructors
  AeroComCldTop(const ekat::Comm &comm, const ekat::ParameterList &params);

  // The name of the diagnostic
  std::string name() const override { return "AeroComCldTop"; }

  // Set the grid
  void set_grids(
      const std::shared_ptr<const GridsManager> grids_manager) override;

 protected:
#ifdef KOKKOS_ENABLE_CUDA
 public:
#endif
  void compute_diagnostic_impl();

  int m_ncols;
  int m_nlevs;
  int m_ndiag;
};

}  // namespace scream

#endif  // EAMXX_AEROCOMCLDTOP_DIAG
