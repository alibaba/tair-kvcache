# Automatic quota support

The storage backends shipped in the open-source build currently support fixed quotas only.
Support for automatic quotas will be added backend by backend. A backend can opt in by
implementing `SupportsStorageCapacity()` and returning a cached capacity sample from
`GetStorageCapacity()`.
