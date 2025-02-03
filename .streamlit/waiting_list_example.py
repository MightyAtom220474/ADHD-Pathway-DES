def prefill_queues(self):
        """
        Method to pre fill queues with set numbers that are defined in the model class

        First fills the patients who have already had their appointments with the surgical clinic
        This ensures they get the earliest/lowest patient IDs

        Next, prefill with those patients who have not yet had their surgical clinic appointments
        so will need to proceed through the whole pathway
        """

        log.debug(f"Prefilling admitted queues with {self.fill_admitted_queue} patients")

        # Fill admitted queue
        for i in range(self.fill_admitted_queue):

            # Increment patient counter by 1
            self.patient_counter += 1
            self.active_entities += 1

            # Create new patient
            pt = Patient(self.patient_counter)

            # Note that unlike the patients in the 'non-admitted' queue,
            # these patients have a value of 'already_seen_clinic' of true,
            # so will begin at an earlier point in the pathway
            pt.already_seen_clinic = True
            pt.from_prefills = True

            self.event_log.append(
                {'patient': self.patient_counter, 'event_type': 'arrival_departure',
                 'event': 'arrival', 'time': self.env.now,
                 'prefill': pt.from_prefills, 'prefill_already_seen_clinic': pt.already_seen_clinic,
                 'before_end_sim': pt.before_end_sim, 'surgery_required': pt.needs_surgery
                 }
            )

            # Get simpy env to run enter_pathway method with this patient
            self.env.process(self.enter_pathway(pt))

            # Need to have yield statement so code works - timeout for zero time
            yield self.env.timeout(0)

        log.debug(f"Prefilling non-admitted queues with {self.fill_non_admitted_queue} patients")

        # Fill non-admitted queue
        for i in range(self.fill_non_admitted_queue):

            # Increment patient counter by 1
            self.patient_counter += 1
            self.active_entities += 1

            # Create new patient
            pt = Patient(self.patient_counter)
            pt.from_prefills = True
            self.event_log.append(
                {'patient': self.patient_counter, 'event_type': 'arrival_departure',
                 'event': 'arrival', 'time': self.env.now,
                 'prefill': pt.from_prefills, 'prefill_already_seen_clinic': pt.already_seen_clinic,
                 'before_end_sim': pt.before_end_sim, 'surgery_required': pt.needs_surgery
                 }
            )
            # NOTE: the default value for new patients is that they have not
            # already been seen by the clinic
            # So these patients in the non-admitted queue have
            # the attribute *already_seen_clinic*=False

            # Get simpy env to run enter_pathway method with this patient
            self.env.process(self.enter_pathway(pt))
            yield self.env.timeout(0)