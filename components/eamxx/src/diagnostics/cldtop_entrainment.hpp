#ifndef EAMXX_CLDTOP_ENTRAINMENT_HPP
#define EAMXX_CLDTOP_ENTRAINMENT_HPP

#include "share/atm_process/atmosphere_diagnostic.hpp"

namespace scream {

/*
 * This diagnostic will compute cloud-top entrainment diagnostics.
 */

class CldtopEntrainment : public AtmosphereDiagnostic {
 public:
  // Constructors
  CldtopEntrainment(const ekat::Comm &comm, const ekat::ParameterList &params);

  // The name of the diagnostic
  std::string name() const override { return "cldtop_entrainment"; }

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

  // How many outputs do we want?
  int m_num_outputs = 1;
};

}  // namespace scream

#endif  // EAMXX_CLDTOP_ENTRAINMENT_HPP
