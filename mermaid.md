sequenceDiagram
    autonumber
    actor Dev as Developer
    participant PR as Pull Request
    participant Gov as pr-governance.yml
    participant Qlt as pr-quality.yml
    participant Sec as pr-security.yml
    participant Img as image-security.yml
    participant Files as Governance Files
    participant Rev as Reviewer / Merge Gate

    rect rgb(255, 248, 196)
        Note over Dev,PR: Phase 1 - Open / Update PR
        Dev->>PR: Push branch and create / update PR
        Dev->>PR: Fill in title and PR template
    end

    rect rgb(227, 242, 253)
        Note over PR,Gov: Phase 2 - Governance Checks
        PR->>Gov: Trigger governance workflow
        Files-->>Gov: CODEOWNERS / policy files
        Gov->>Gov: Validate branch name
        Gov->>Gov: Validate PR title
        Gov->>Gov: Validate required template fields
        alt Sensitive paths were changed
            Gov-->>PR: Require matching owner / approval
        else No sensitive paths changed
            Gov-->>PR: Governance passed
        end
    end

    rect rgb(232, 245, 233)
        Note over PR,Qlt: Phase 3 - Quality Checks
        PR->>Qlt: Trigger quality workflow
        Qlt->>Qlt: Lint
        Qlt->>Qlt: Unit tests
        Qlt->>Qlt: Coverage threshold
        Qlt->>Qlt: Build
        Qlt->>Qlt: SonarQube analysis
        Qlt-->>PR: Quality result
    end

    rect rgb(255, 243, 224)
        Note over PR,Img: Phase 4 - Security Checks
        PR->>Sec: Trigger security workflow
        Files-->>Sec: SECURITY.md / .gitleaks.toml
        Sec->>Sec: SAST via SonarQube
        Sec->>Sec: SCA via Trivy
        Sec->>Sec: Secret scan via Trivy / Gitleaks
        Sec->>Sec: Export SARIF
        alt Container image is built
            Sec->>Img: Trigger image-security
            Img->>Img: Build image tag based on SHA
            Img->>Img: Trivy image scan
            Img->>Img: Fail based on severity policy
            Img->>Img: Upload SARIF / artifact
            Img->>Img: Optional SBOM generation
            Img-->>PR: Image security result
        else No image
            Sec-->>PR: Skip image-security
        end
        Sec-->>PR: Security result
    end

    rect rgb(243, 229, 245)
        Note over PR,Rev: Phase 5 - Review / Merge
        PR->>Rev: Wait for all checks to pass
        Rev-->>PR: Review / approval
        Rev-->>Dev: Merge allowed
    end
