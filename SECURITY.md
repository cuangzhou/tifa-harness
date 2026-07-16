# Security policy

Report suspected vulnerabilities privately through GitHub Security Advisories. Do not include credentials or private repository contents in a report.

Docker execution is the supported strong-isolation boundary. The default policy denies networking, drops capabilities, uses a non-root user and read-only root filesystem, and applies CPU, memory, PID, and timeout limits. Local execution is explicitly degraded and should only be selected for trusted commands.

Tifa redacts credential-shaped fields from persisted artifacts. Users remain responsible for workspace permissions, Docker daemon access, image provenance, and provider-account controls.
