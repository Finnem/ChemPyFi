# Vendored Dependencies

This directory contains third-party packages bundled with chempyfi to ensure compatibility and ease of installation.

## ⚠️ IMPORTANT

**These packages are NOT developed or maintained by the chempyfi team.**

This is a "vendoring" approach where we include a copy of dependencies directly in our source tree to:
- Avoid complex installation requirements
- Prevent version conflicts
- Ensure reproducibility
- Work around path configuration issues

## Included Packages

### ledaw_package

**Purpose:** LED (Local Energy Decomposition) Analysis Workflow for ORCA quantum chemistry calculations

**Original Source:** LEDAW Development Team
- Official repository/contact: (Add if known)
- Included version: (Check ledaw_package for version info)

**Why Vendored:**
- Complex installation requirements (compiled extensions)
- Path configuration issues causing segmentation faults
- Required for LED extraction functionality in `chempyfi.orcautil`

**License:** See `ledaw_package/` directory for original license information

**Modifications:** None - included as-is from original distribution

**Attribution:** 
LEDAW is developed and maintained by its original authors. All credit for LEDAW functionality belongs to the LEDAW development team. This is a bundled copy for convenience only.

## For Developers

### Updating Vendored Packages

To update a vendored package:

1. Obtain the latest version from the original source
2. Test thoroughly with chempyfi functionality
3. Replace the directory contents
4. Update version information in this README
5. Document any changes needed in chempyfi code

### Adding New Vendored Packages

Only vendor packages when absolutely necessary:
- Complex installation requirements
- No pip/conda package available
- Critical for core functionality
- Benefits outweigh maintenance burden

Document thoroughly:
- Original source and license
- Reason for vendoring
- Version information
- Any modifications made

## Legal

All vendored packages remain under their original licenses. chempyfi's license does NOT apply to these vendored dependencies. See individual package directories for license information.




