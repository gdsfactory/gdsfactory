CREATE DATABASE  IF NOT EXISTS `database`;
USE `database`;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;

--
-- Table structure for table `circuit`
--

DROP TABLE IF EXISTS `circuit`;
CREATE TABLE `circuit` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `die_id` int NOT NULL,
  `name` varchar(250) NOT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_circuit_die_id_idx` (`die_id`),
  CONSTRAINT `fk_circuit_die_id` FOREIGN KEY (`die_id`) REFERENCES `die` (`id`)
) COMMENT='This table holds the definition of circuits.';

--
-- Table structure for table `component_info`
--

DROP TABLE IF EXISTS `component_info`;
CREATE TABLE `component_info` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `circuit_id` int,
  `die_id` int,
  `port_id` int,
  `reticle_id` int,
  `wafer_id` int,
  `name` varchar(200) NOT NULL,
  `value` varchar(200) NOT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_ci_circuit_id_idx` (`circuit_id`),
  KEY `fk_ci_die_id_idx` (`die_id`),
  KEY `fk_ci_port_id_idx` (`port_id`),
  KEY `fk_ci_reticle_id_idx` (`reticle_id`),
  KEY `fk_ci_wafer_id_idx` (`wafer_id`),
  CONSTRAINT `fk_ci_circuit_id` FOREIGN KEY (`circuit_id`) REFERENCES `circuit` (`id`),
  CONSTRAINT `fk_ci_die_id` FOREIGN KEY (`die_id`) REFERENCES `die` (`id`),
  CONSTRAINT `fk_ci_port_id` FOREIGN KEY (`port_id`) REFERENCES `port` (`id`),
  CONSTRAINT `fk_ci_reticle_id` FOREIGN KEY (`reticle_id`) REFERENCES `reticle` (`id`),
  CONSTRAINT `fk_ci_wafer_id` FOREIGN KEY (`wafer_id`) REFERENCES `wafer` (`id`)
) COMMENT='This table holds information for the component using name/value pairs with optional description.';

--
-- Table structure for table `computed_result`
--

DROP TABLE IF EXISTS `computed_result`;
CREATE TABLE `computed_result` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `name` varchar(200) NOT NULL,
  `type` varchar(50) NOT NULL,
  `unit_id` int DEFAULT NULL,
  `domain_unit_id` int DEFAULT NULL,
  `value` longtext NOT NULL,
  `domain` longtext,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_cr_unit_id_idx` (`unit_id`),
  KEY `fk_cr_domain_unit_id_idx` (`domain_unit_id`),
  CONSTRAINT `fk_cr_domain_unit_id` FOREIGN KEY (`domain_unit_id`) REFERENCES `unit` (`id`),
  CONSTRAINT `fk_cr_unit_id` FOREIGN KEY (`unit_id`) REFERENCES `unit` (`id`)
) COMMENT='This table holds all results obtained after computation/analysis of the raw results contained in the table result.';

--
-- Table structure for table `result_self_relation`
--

DROP TABLE IF EXISTS `computed_result_self_relation`;
CREATE TABLE `computed_result_self_relation` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `computed_result1_id` int NOT NULL,
  `computed_result2_id` int NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_crsr_computed_result1_id_idx` (`computed_result1_id`),
  KEY `fk_crsr_computed_result2_id_idx` (`computed_result2_id`),
  CONSTRAINT `fk_crsr_computed_result1_id` FOREIGN KEY (`computed_result1_id`) REFERENCES `computed_result` (`id`),
  CONSTRAINT `fk_crsr_computed_result2_id` FOREIGN KEY (`computed_result2_id`) REFERENCES `computed_result` (`id`)
) COMMENT='This table holds all computed results self relation. This is used to link computed results together';

--
-- Table structure for table `die`
--

DROP TABLE IF EXISTS `die`;
CREATE TABLE `die` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `reticle_id` int NOT NULL,
  `name` varchar(200) NOT NULL,
  `position` varchar(50) DEFAULT NULL,
  `size` varchar(50) DEFAULT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_die_reticle_id_idx` (`reticle_id`),
  CONSTRAINT `fk_die_reticle_id` FOREIGN KEY (`reticle_id`) REFERENCES `reticle` (`id`)
) COMMENT='This table holds die definition.';

--
-- Table structure for table `port`
--

DROP TABLE IF EXISTS `port`;
CREATE TABLE `port` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `circuit_id` int NOT NULL,
  `name` varchar(200) DEFAULT '',
  `is_electrical` tinyint NOT NULL COMMENT 'Boolean. if the port is electrical.',
  `is_optical` tinyint NOT NULL COMMENT 'Boolean. if the port is optical.',
  `position` varchar(50) NOT NULL,
  `orientation` double NOT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_ep_circuit_id_idx` (`circuit_id`),
  CONSTRAINT `fk_ep_circuit_id` FOREIGN KEY (`circuit_id`) REFERENCES `circuit` (`id`)
) COMMENT='This table holds all ports definition.';

--
-- Table structure for table `relation_info`
--

DROP TABLE IF EXISTS `relation_info`;
CREATE TABLE `relation_info` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `computed_result_self_relation_id` int,
  `result_self_relation_id` int,
  `result_process_relation_id` int,
  `result_component_relation_id` int,
  `result_computed_result_relation_id` int,
  `name` varchar(200) NOT NULL,
  `value` mediumtext NOT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_ri_computed_result_self_relation_id_idx` (`computed_result_self_relation_id`),
  KEY `fk_ri_result_self_relation_id_idx` (`result_self_relation_id`),
  KEY `fk_ri_result_process_relation_id_idx` (`result_process_relation_id`),
  KEY `fk_ri_result_component_relation_id_idx` (`result_component_relation_id`),
  KEY `fk_ri_result_computed_result_relation_id_idx` (`result_computed_result_relation_id`),
  CONSTRAINT `fk_ri_computed_result_self_relation_id` FOREIGN KEY (`computed_result_self_relation_id`) REFERENCES `computed_result_self_relation` (`id`),
  CONSTRAINT `fk_ri_result_self_relation_id` FOREIGN KEY (`result_self_relation_id`) REFERENCES `result_self_relation` (`id`),
  CONSTRAINT `fk_ri_result_process_relation_id` FOREIGN KEY (`result_process_relation_id`) REFERENCES `result_process_relation` (`id`),
  CONSTRAINT `fk_ri_result_component_relation_id` FOREIGN KEY (`result_component_relation_id`) REFERENCES `result_component_relation` (`id`),
  CONSTRAINT `fk_ri_result_computed_result_relation_id` FOREIGN KEY (`result_computed_result_relation_id`) REFERENCES `result_computed_result_relation` (`id`)
) COMMENT='This table holds extra information about specific relation.';

--
-- Table structure for table `result`
--

DROP TABLE IF EXISTS `result`;
CREATE TABLE `result` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `name` varchar(200) NOT NULL,
  `type` varchar(50) NOT NULL,
  `unit_id` int DEFAULT NULL,
  `domain_unit_id` int DEFAULT NULL,
  `value` longtext NOT NULL,
  `domain` longtext,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_r_unit_id_idx` (`unit_id`),
  KEY `fk_r_domain_unit_id_idx` (`domain_unit_id`),
  CONSTRAINT `fk_r_unit_id` FOREIGN KEY (`unit_id`) REFERENCES `unit` (`id`),
  CONSTRAINT `fk_r_domain_unit_id` FOREIGN KEY (`domain_unit_id`) REFERENCES `unit` (`id`)
) COMMENT='This table holds all results.';

--
-- Table structure for table `result_component_relation`
--

DROP TABLE IF EXISTS `result_component_relation`;
CREATE TABLE `result_component_relation` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `result_id` int NOT NULL,
  `circuit_id` int,
  `die_id` int,
  `port_id` int,
  `reticle_id` int,
  `wafer_id` int,
  PRIMARY KEY (`id`),
  KEY `fk_rcr_result_id_idx` (`result_id`),
  KEY `fk_rcr_circuit_id_idx` (`circuit_id`),
  KEY `fk_rcr_die_id_idx` (`die_id`),
  KEY `fk_rcr_port_id_idx` (`port_id`),
  KEY `fk_rcr_reticle_id_idx` (`reticle_id`),
  KEY `fk_rcr_wafer_id_idx` (`wafer_id`),
  CONSTRAINT `fk_rcr_circuit_id` FOREIGN KEY (`circuit_id`) REFERENCES `circuit` (`id`),
  CONSTRAINT `fk_rcr_die_id` FOREIGN KEY (`die_id`) REFERENCES `die` (`id`),
  CONSTRAINT `fk_rcr_port_id` FOREIGN KEY (`port_id`) REFERENCES `port` (`id`),
  CONSTRAINT `fk_rcr_reticle_id` FOREIGN KEY (`reticle_id`) REFERENCES `reticle` (`id`),
  CONSTRAINT `fk_rcr_wafer_id` FOREIGN KEY (`wafer_id`) REFERENCES `wafer` (`id`),
  CONSTRAINT `fk_rcr_result_id` FOREIGN KEY (`result_id`) REFERENCES `result` (`id`)
) COMMENT='This table holds the relations in between results and components.';

--
-- Table structure for table `result_computed_result_relation`
--

DROP TABLE IF EXISTS `result_computed_result_relation`;
CREATE TABLE `result_computed_result_relation` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `result_id` int NOT NULL,
  `computed_result_id` int NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_rcrr_result_id_idx` (`result_id`),
  KEY `fk_rcrr_computed_result_id_idx` (`computed_result_id`),
  CONSTRAINT `fk_rcrr_computed_result_id` FOREIGN KEY (`computed_result_id`) REFERENCES `computed_result` (`id`),
  CONSTRAINT `fk_rcrr_result_id` FOREIGN KEY (`result_id`) REFERENCES `result` (`id`)
) COMMENT='This table holds the relations in between the results and the computed results.';

--
-- Table structure for table `result_info`
--

DROP TABLE IF EXISTS `result_info`;
CREATE TABLE `result_info` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `name` varchar(200) NOT NULL,
  `value` mediumtext NOT NULL,
  `result_id` int NOT NULL,
  `computed_result_id` int NOT NULL,
  `unit_id` int DEFAULT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_ri_result_id_idx` (`result_id`),
  KEY `fk_ri_computed_result_id_idx` (`computed_result_id`),
  KEY `fk_ri_unit_id_idx` (`unit_id`),
  CONSTRAINT `fk_ri_result_id` FOREIGN KEY (`result_id`) REFERENCES `result` (`id`),
  CONSTRAINT `fk_ri_computed_result_id` FOREIGN KEY (`computed_result_id`) REFERENCES `computed_result` (`id`),
  CONSTRAINT `fk_ti_unit_id` FOREIGN KEY (`unit_id`) REFERENCES `unit` (`id`)
) COMMENT='This table holds extra information about specific results.';

--
-- Table structure for table `result_self_relation`
--

DROP TABLE IF EXISTS `result_self_relation`;
CREATE TABLE `result_self_relation` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `result1_id` int NOT NULL,
  `result2_id` int NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_rsr_result1_id_idx` (`result1_id`),
  KEY `fk_rsr_result2_id_idx` (`result2_id`),
  CONSTRAINT `fk_rsr_result1_id` FOREIGN KEY (`result1_id`) REFERENCES `result` (`id`),
  CONSTRAINT `fk_rsr_result2_id` FOREIGN KEY (`result2_id`) REFERENCES `result` (`id`)
) COMMENT='This table holds all results self relation. This is used to link results together';

--
-- Table structure for table `result_process_relation`
--

DROP TABLE IF EXISTS `result_process_relation`;
CREATE TABLE `result_process_relation` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `result_id` int NOT NULL,
  `process_id` int NOT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_rpr_result_id_idx` (`result_id`),
  KEY `fk_rpr_process_id_idx` (`process_id`),
  CONSTRAINT `fk_rpr_result_id` FOREIGN KEY (`result_id`) REFERENCES `result` (`id`),
  CONSTRAINT `fk_rpr_process_id` FOREIGN KEY (`process_id`) REFERENCES `process` (`id`)
) COMMENT='This table holds all results and simulation result relation.';

--
-- Table structure for table `reticle`
--

DROP TABLE IF EXISTS `reticle`;
CREATE TABLE `reticle` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `name` varchar(200) NOT NULL,
  `position` varchar(50) DEFAULT NULL COMMENT 'Position of the reticle on the wafer. (ROW, COLUMN)',
  `size` varchar(50) DEFAULT NULL COMMENT 'The size of the reticle (X,Y) having the convention that -Å· points towards the notch/flat of the wafer.',
  `wafer_id` int NOT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`),
  KEY `fk_reticle_wafer_id_idx` (`wafer_id`),
  CONSTRAINT `fk_reticle_wafer_id` FOREIGN KEY (`wafer_id`) REFERENCES `wafer` (`id`)
) COMMENT='This table holds the definition of a reticle.'; /*Further reticle information are contained in the table reticle_info.*/

--
-- Table structure for table `process`
--

DROP TABLE IF EXISTS `process`;
CREATE TABLE `process` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `name` varchar(200) NOT NULL,
  `process` longblob NOT NULL,
  `version` varchar(50) NOT NULL,
  `type` varchar(50),
  `description` mediumtext,
  PRIMARY KEY (`id`)
) COMMENT='This table holds all simulation results.';

--
-- Table structure for table `unit`
--

DROP TABLE IF EXISTS `unit`;
CREATE TABLE `unit` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `name` varchar(200) NOT NULL,
  `quantity` varchar(200) NOT NULL,
  `symbol` varchar(50) NOT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`)
) COMMENT='This table holds all units. A unit is here understood as definite magnitude of a quantity.';

--
-- Table structure for table `wafer`
--

DROP TABLE IF EXISTS `wafer`;
CREATE TABLE `wafer` (
  `id` int NOT NULL AUTO_INCREMENT,
  `created` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `updated` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `serial_number` varchar(200) NOT NULL,
  `name` varchar(200) DEFAULT NULL,
  `description` mediumtext,
  PRIMARY KEY (`id`)
) COMMENT='This table holds the base definition of a wafer.'; /*Further information is logged in table wafer_info.*/

/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
