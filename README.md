
# Market Mimic: Synthetic Tick Data Generation with GANs

## Overview
Market Mimic is an innovative project that uses Generative Adversarial Networks (GANs) to generate synthetic tick data emulating financial markets. This tool aims to assist financial analysts, quantitative researchers, and algorithmic traders by providing them with high-fidelity data for backtesting and market simulation purposes.

## Features
- **Synthetic Tick Data Generation**: Generate realistic tick-level market data.
- **Customizable GAN Models**: Flexibility to adjust models according to specific market conditions or data types.
- **Evaluation Tools**: Includes tools to evaluate the realism and quality of the generated data against actual market data.

## Installation

### Setting Up the Environment
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/joaquinbejar/MarketMimic.git
cd MarketMimic
make create-venv
source venv/bin/activate
```

### Installing Dependencies
Install all necessary dependencies using the make command:

```bash
make install-dep
```

## Usage

### Building the Project
To build the project and prepare it for distribution:

```bash
make build
```

### Running Tests
You can run different types of tests with these commands:

- **Unit Tests**:
  ```bash
  make run-unit-tests
  ```

- **Extended Tests**:
  ```bash
  make run-extended-tests
  ```

- **Unit Test Coverage**:
  ```bash
  make run-unit-test-coverage
  ```

### Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

### Versioning

We use [SemVer](http://semver.org/) for versioning. This approach allows us to maintain a clear, predictable system for version management. Under this scheme, version numbers are given in the format of `MAJOR.MINOR.PATCH`, where:

- `MAJOR` versions indicate incompatible API changes,
- `MINOR` versions add functionality in a backwards-compatible manner, and
- `PATCH` versions include backwards-compatible bug fixes.

This standard helps users and developers to understand the impact of new updates at a glance. For the versions available, see the [tags on this repository](https://github.com/joaquinbejar/IronFix/tags).

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
- Thanks to all the contributors who spend time to help improve this project.
- Special thanks to financial market data providers for making their data available for analysis.

## Support
For support, email fake-email@domain.com or join our Slack channel.

### Authors
**Joaquín Béjar García** - Initial work - [joaquinbejar](https://github.com/joaquinbejar)

See also the list of contributors who participated in this project.