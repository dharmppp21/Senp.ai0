pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}
rootProject.name = "Senp.ai0"
include(
    ":core-contracts",
    ":core-pipeline",
    ":core-motion",
    ":core-alignment",
    ":core-cache",
    ":headless-runner",
    ":android-video",
    ":android-pose-mediapipe",
    ":validation-app",
    ":sync-v2-integration",
    ":sync-v2-validation",
)
